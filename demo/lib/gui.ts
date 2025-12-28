import GUI from 'lil-gui';

const gui = new GUI();

const params: { type: 'arcball' | 'WASD' } = {
	type: 'arcball',
};

interface Controls {
	pause: boolean;
	cameraType: 'arcball' | 'WASD';
	highDPI: boolean;
	bunnyRotation: number;
}

const controls: Controls = {
	pause: false,
	cameraType: 'arcball',
	highDPI: false,
	bunnyRotation: 0.0,
};

const pauseController = gui.add(controls, 'pause').name('Pause');
const cameraController = gui.add(controls, 'cameraType', ['arcball', 'WASD']).name('Camera Type');
const highDPIController = gui.add(controls, 'highDPI').name('High DPI');
const rotationController = gui.add(controls, 'bunnyRotation', 0, 360, 1).name('Bunny Rotation');

export { controls, pauseController, cameraController, highDPIController, rotationController };
export type { Controls };

export default gui;
