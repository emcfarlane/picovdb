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
}

export interface GUIControllers {
	controls: Controls;
	modelController: ReturnType<GUI['add']>;
	pauseController: ReturnType<GUI['add']>;
	cameraController: ReturnType<GUI['add']>;
	highDPIController: ReturnType<GUI['add']>;
	rotationController: ReturnType<GUI['add']>;
	debugController: ReturnType<GUI['add']>;
}

export function initGUI(models: ModelConfig[]): GUIControllers {
	const gui = new GUI();

	const controls: Controls = {
		pause: false,
		highDPI: false,
		rotation: 0.0,
		resetCamera: () => { },
		debugIterations: false,
		model: models[0].name,
	};

	const modelController = gui.add(controls, 'model', models.map(m => m.name)).name('Model');
	const pauseController = gui.add(controls, 'pause').name('Pause');
	const cameraController = gui.add(controls, 'resetCamera').name('Reset Camera');
	const highDPIController = gui.add(controls, 'highDPI').name('High DPI');
	const rotationController = gui.add(controls, 'rotation', 0, 360, 1).name('Rotation');
	const debugController = gui.add(controls, 'debugIterations').name('Debug Iterations');

	return { controls, modelController, pauseController, cameraController, highDPIController, rotationController, debugController };
}
