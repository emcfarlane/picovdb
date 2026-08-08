const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add OpenVDB dependency for NanoVDB headers
    const openvdb = b.dependency("openvdb", .{});

    const mod = b.addModule("picovdb", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "picovdb",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "picovdb", .module = mod },
            },
        }),
    });

    // Configure NanoVDB headers
    exe.root_module.addIncludePath(openvdb.path("nanovdb/nanovdb"));
    exe.root_module.link_libc = true;

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // Add NanoVDB headers and C library for tests
    exe_tests.root_module.addIncludePath(openvdb.path("nanovdb/nanovdb"));
    exe_tests.root_module.link_libc = true;

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const c_api_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true, // c_api uses std.heap.c_allocator on native
            .imports = &.{
                .{ .name = "picovdb", .module = mod },
            },
        }),
    });
    const run_c_api_tests = b.addRunArtifact(c_api_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_c_api_tests.step);

    // WASM build of the C API (consumed by stl.ts).
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    const wasm = b.addExecutable(.{
        .name = "picovdb",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = wasm_target,
            .optimize = .ReleaseSmall,
            .single_threaded = true, // std.heap.wasm_allocator requires it
            .imports = &.{
                .{ .name = "picovdb", .module = b.createModule(.{
                    .root_source_file = b.path("src/root.zig"),
                    .target = wasm_target,
                    .optimize = .ReleaseSmall,
                }) },
            },
        }),
    });
    wasm.entry = .disabled; // library, no _start
    wasm.rdynamic = true; // keep `export fn` symbols
    const wasm_step = b.step("wasm", "Build the WASM C API module");
    wasm_step.dependOn(&b.addInstallArtifact(wasm, .{
        .dest_dir = .{ .override = .{ .custom = "wasm" } },
    }).step);

    // Apple static libs -> PicoVDB.xcframework (macOS host only).
    const xc_out = b.getInstallPath(.prefix, "PicoVDB.xcframework");
    const xc_rm = b.addSystemCommand(&.{ "rm", "-rf", xc_out });
    const xcodebuild = b.addSystemCommand(&.{ "xcodebuild", "-create-xcframework" });
    xcodebuild.step.dependOn(&xc_rm.step);
    const apple_slices = [_]std.Target.Query{
        // The `zig build-lib -target aarch64-ios` CLI mis-tags Mach-O as macOS;
        // an explicit os_version_min here makes LC_BUILD_VERSION correct.
        .{ .cpu_arch = .aarch64, .os_tag = .ios, .abi = .none, .os_version_min = .{ .semver = .{ .major = 16, .minor = 0, .patch = 0 } } },
        .{ .cpu_arch = .aarch64, .os_tag = .ios, .abi = .simulator, .os_version_min = .{ .semver = .{ .major = 16, .minor = 0, .patch = 0 } } },
        .{ .cpu_arch = .aarch64, .os_tag = .macos, .os_version_min = .{ .semver = .{ .major = 13, .minor = 0, .patch = 0 } } },
    };
    for (apple_slices) |query| {
        const slice_target = b.resolveTargetQuery(query);
        // Emit an object and archive it with Apple's libtool. Zig's own
        // archiver pads members to 2 bytes; Apple's ld requires Mach-O members
        // at 8-byte offsets, so zig-made archives fail to link depending on
        // symbol-table size and member-name length.
        const obj = b.addObject(.{
            .name = "picovdb",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/c_api.zig"),
                .target = slice_target,
                .optimize = .ReleaseFast,
                .link_libc = true,
                .imports = &.{
                    .{ .name = "picovdb", .module = b.createModule(.{
                        .root_source_file = b.path("src/root.zig"),
                        .target = slice_target,
                        .optimize = .ReleaseFast,
                    }) },
                },
            }),
        });
        const libtool = b.addSystemCommand(&.{ "libtool", "-static", "-o" });
        const lib = libtool.addOutputFileArg("libpicovdb.a");
        libtool.addFileArg(obj.getEmittedBin());
        xcodebuild.addArg("-library");
        xcodebuild.addFileArg(lib);
        xcodebuild.addArg("-headers");
        xcodebuild.addDirectoryArg(b.path("include"));
    }
    xcodebuild.addArgs(&.{ "-output", xc_out });
    const xc_step = b.step("xcframework", "Build PicoVDB.xcframework (macOS host only)");
    xc_step.dependOn(&xcodebuild.step);
}
