import GUI from 'lil-gui';

export interface ModelConfig {
	name: string;
	url: string;
	translation: [number, number, number];
	scale: number;
}

export interface Controls {
	pause: boolean;
	highDPI: boolean;
	rotation: number;
	resetCamera: () => void;
	debugIterations: boolean;
	model: string;
	environment: 'Studio' | 'Sky' | 'Studio HDRI';
	whiteBackdrop: boolean;
	lightingMode: 'Reference' | 'ReSTIR';
	denoise: boolean;
	illuminant: string;
	lightIntensity: number;
	domeIntensity: number;
	exposure: number;
	maxBounces: number;
	renderScale: number;
}

export interface GUIControllers {
	controls: Controls;
	gui: GUI;
	modelController: ReturnType<GUI['add']>;
	pauseController: ReturnType<GUI['add']>;
	cameraController: ReturnType<GUI['add']>;
	highDPIController: ReturnType<GUI['add']>;
	rotationController: ReturnType<GUI['add']>;
	debugController: ReturnType<GUI['add']>;
	illuminantController: ReturnType<GUI['add']>;
}

export function initGUI(models: ModelConfig[], illuminants: string[], initialModelName?: string): GUIControllers {
	const gui = new GUI();

	const controls: Controls = {
		pause: false,
		highDPI: false,
		rotation: 0.0,
		resetCamera: () => { },
		debugIterations: false,
		model: initialModelName ?? models[0].name,
		environment: 'Studio HDRI',
		whiteBackdrop: true,
		lightingMode: 'Reference',
		denoise: true,
		illuminant: illuminants[0],
		lightIntensity: 15.0,
		domeIntensity: 0.5,
		exposure: 1.0,
		maxBounces: 4,
		renderScale: 0.5,
		// Temporal + paired-spatial ReSTIR DI: measured at parity with the
		// path tracer's 3-technique NEE on this scene while spending 1/3 of
		// its direct shadow rays; pulls ahead as lights multiply
	};

	const modelController = gui.add(controls, 'model', models.map(m => m.name)).name('Model');
	const pauseController = gui.add(controls, 'pause').name('Pause');
	const cameraController = gui.add(controls, 'resetCamera').name('Reset Camera');
	const highDPIController = gui.add(controls, 'highDPI').name('High DPI');
	const rotationController = gui.add(controls, 'rotation', 0, 360, 1).name('Rotation');
	const debugController = gui.add(controls, 'debugIterations').name('Debug Iterations');

	const lighting = gui.addFolder('Lighting');
	// ReSTIR = ReSTIR PT (docs/restir-pt-plan.md); Reference = ground truth
	lighting.add(controls, 'lightingMode', ['Reference', 'ReSTIR']).name('Lighting');
	// SVGF on the ReSTIR output (temporal + a-trous); off = raw accumulation
	lighting.add(controls, 'denoise').name('Denoise');
	lighting.add(controls, 'environment', ['Studio HDRI', 'Studio', 'Sky']).name('Environment');
	lighting.add(controls, 'whiteBackdrop').name('White Backdrop');
	const illuminantController = lighting.add(controls, 'illuminant', illuminants).name('Illuminant');
	lighting.add(controls, 'domeIntensity', 0, 4, 0.05).name('Dome Intensity');
	lighting.add(controls, 'exposure', 0.01, 4, 0.01).name('Exposure');
	lighting.add(controls, 'maxBounces', 1, 8, 1).name('Max Bounces');
	lighting.add(controls, 'renderScale', { '25%': 0.25, '50%': 0.5, '100%': 1 }).name('Render Scale');
	// ReSTIR DI for direct lighting; off = pure path-traced reference

	return { controls, gui, modelController, pauseController, cameraController, highDPIController, rotationController, debugController, illuminantController };
}
