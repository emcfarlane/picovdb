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

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
