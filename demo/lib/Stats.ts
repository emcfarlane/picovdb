// stats based on stats.js

export class Panel {
	private canvas: HTMLCanvasElement;
	private context: CanvasRenderingContext2D;
	private name: string;
	private fg: string;
	private bg: string;
	private min = Infinity;
	private max = 0;
	private round = Math.round;
	private PR = Math.round(window.devicePixelRatio || 1);

	private WIDTH = 80 * this.PR;
	private HEIGHT = 48 * this.PR;
	private TEXT_X = 3 * this.PR;
	private TEXT_Y = 2 * this.PR;
	private GRAPH_X = 3 * this.PR;
	private GRAPH_Y = 15 * this.PR;
	private GRAPH_WIDTH = 74 * this.PR;
	private GRAPH_HEIGHT = 30 * this.PR;

	constructor(name: string, fg: string, bg: string) {
		this.name = name;
		this.fg = fg;
		this.bg = bg;

		this.canvas = document.createElement('canvas');
		this.canvas.width = this.WIDTH;
		this.canvas.height = this.HEIGHT;
		this.canvas.style.cssText = 'width:80px;height:48px';

		this.context = this.canvas.getContext('2d')!;
		this.context.font = 'bold ' + (9 * this.PR) + 'px Helvetica,Arial,sans-serif';
		this.context.textBaseline = 'top';

		this.context.fillStyle = bg;
		this.context.fillRect(0, 0, this.WIDTH, this.HEIGHT);

		this.context.fillStyle = fg;
		this.context.fillText(name, this.TEXT_X, this.TEXT_Y);
		this.context.fillRect(this.GRAPH_X, this.GRAPH_Y, this.GRAPH_WIDTH, this.GRAPH_HEIGHT);

		this.context.fillStyle = bg;
		this.context.globalAlpha = 0.9;
		this.context.fillRect(this.GRAPH_X, this.GRAPH_Y, this.GRAPH_WIDTH, this.GRAPH_HEIGHT);
	}

	update(value: number, maxValue: number): void {
		this.min = Math.min(this.min, value);
		this.max = Math.max(this.max, value);

		this.context.fillStyle = this.bg;
		this.context.globalAlpha = 1;
		this.context.fillRect(0, 0, this.WIDTH, this.GRAPH_Y);
		this.context.fillStyle = this.fg;
		this.context.fillText(
			this.round(value) + ' ' + this.name + ' (' + this.round(this.min) + '-' + this.round(this.max) + ')',
			this.TEXT_X,
			this.TEXT_Y
		);

		this.context.drawImage(
			this.canvas,
			this.GRAPH_X + this.PR,
			this.GRAPH_Y,
			this.GRAPH_WIDTH - this.PR,
			this.GRAPH_HEIGHT,
			this.GRAPH_X,
			this.GRAPH_Y,
			this.GRAPH_WIDTH - this.PR,
			this.GRAPH_HEIGHT
		);

		this.context.fillRect(this.GRAPH_X + this.GRAPH_WIDTH - this.PR, this.GRAPH_Y, this.PR, this.GRAPH_HEIGHT);

		this.context.fillStyle = this.bg;
		this.context.globalAlpha = 0.9;
		this.context.fillRect(
			this.GRAPH_X + this.GRAPH_WIDTH - this.PR,
			this.GRAPH_Y,
			this.PR,
			this.round((1 - value / maxValue) * this.GRAPH_HEIGHT)
		);
	}

	get dom(): HTMLCanvasElement {
		return this.canvas;
	}
}

export class Stats {
	private container: HTMLDivElement;
	private fpsPanel: Panel;
	//private msPanel: Panel;
	private memPanel: Panel | null = null;

	private beginTime = (performance || Date).now();
	private prevTime = this.beginTime;
	private frames = 0;

	constructor() {
		this.container = document.createElement('div');
		this.container.style.cssText = 'position:fixed;top:0;left:0;opacity:0.9;z-index:10000';

		// FPS Panel
		this.fpsPanel = this.addPanel(new Panel('FPS', '#0ff', '#002'));
		//// MS Panel  
		//this.msPanel = this.addPanel(new Panel('MS', '#0f0', '#020'));

		// Memory Panel (if supported)
		if ((performance as any).memory) {
			this.memPanel = this.addPanel(new Panel('MB', '#f08', '#201'));
		}
	}

	addPanel(panel: Panel): Panel {
		this.container.appendChild(panel.dom);
		return panel;
	}

	begin(): number {
		this.beginTime = (performance || Date).now();
		return this.beginTime;
	}

	end(): number {
		this.frames++;

		const time = (performance || Date).now();
		const frameTime = time - this.beginTime;

		if (time >= this.prevTime + 1000) {
			console.log(frameTime);
			this.fpsPanel.update((this.frames * 1000) / (time - this.prevTime), 100);

			this.prevTime = time;
			this.frames = 0;

			if (this.memPanel && (performance as any).memory) {
				const memory = (performance as any).memory;
				this.memPanel.update(memory.usedJSHeapSize / 1048576, memory.jsHeapSizeLimit / 1048576);
			}
		}

		return time;
	}

	get dom(): HTMLDivElement {
		return this.container;
	}

	static Panel = Panel;
}
