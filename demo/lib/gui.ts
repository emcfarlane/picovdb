import GUI from 'lil-gui';

const gui = new GUI();

interface Controls {
	pause: boolean;
	highDPI: boolean;
	bunnyRotation: number;
	resetCamera: () => void;
}

const controls: Controls = {
	pause: false,
	highDPI: false,
	bunnyRotation: 0.0,
	resetCamera: () => {},
};

const pauseController = gui.add(controls, 'pause').name('Pause');
const cameraController = gui.add(controls, 'resetCamera').name('Reset Camera');
const highDPIController = gui.add(controls, 'highDPI').name('High DPI');
const rotationController = gui.add(controls, 'bunnyRotation', 0, 360, 1).name('Bunny Rotation');

export { controls, pauseController, cameraController, highDPIController, rotationController };
export type { Controls };

export default gui;
