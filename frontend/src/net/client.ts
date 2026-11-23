import type {
  RunInfo,
  ScenarioSummary,
  StartRunRequest,
  UserSettings,
} from "../generated/protocol";
import { decodeServerMessage, encodeCommand, ProtocolError } from "./protocol";
import type { ClientCommand, ServerMessage } from "./protocol";

export type ConnectionStatus = "idle" | "connecting" | "open" | "reconnecting" | "closed";

/** Exponential backoff with a cap: 250 ms, 500 ms, 1 s, ... up to maxMs. */
export function backoffDelay(attempt: number, baseMs = 250, maxMs = 4000): number {
  return Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
}

export interface RunClientOptions {
  onMessage: (msg: ServerMessage) => void;
  onStatus: (status: ConnectionStatus, detail?: string) => void;
  baseUrl?: string;
  wsFactory?: (url: string) => WebSocket;
  maxAttempts?: number;
}

/** REST helpers plus a WebSocket to one run, reconnecting with backoff if it drops. */
export class RunClient {
  private ws: WebSocket | null = null;
  private wsPath: string | null = null;
  private attempt = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private closedByUser = false;

  constructor(private readonly opts: RunClientOptions) {}

  private url(path: string): string {
    return `${this.opts.baseUrl ?? ""}${path}`;
  }

  async listScenarios(): Promise<ScenarioSummary[]> {
    const res = await fetch(this.url("/api/scenarios"));
    if (!res.ok) throw new Error(`GET /api/scenarios: ${res.status}`);
    return (await res.json()) as ScenarioSummary[];
  }

  async getPreferences(): Promise<UserSettings> {
    const res = await fetch(this.url("/api/preferences"));
    if (!res.ok) throw new Error(`GET /api/preferences: ${res.status}`);
    return (await res.json()) as UserSettings;
  }

  async savePreferences(settings: UserSettings): Promise<UserSettings> {
    const res = await fetch(this.url("/api/preferences"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error(`PUT /api/preferences: ${res.status}`);
    return (await res.json()) as UserSettings;
  }

  async startRun(req: StartRunRequest): Promise<RunInfo> {
    const res = await fetch(this.url("/api/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`POST /api/runs: ${res.status}`);
    const info = (await res.json()) as RunInfo;
    this.connect(info.ws_path);
    return info;
  }

  connect(wsPath: string): void {
    this.close();
    this.closedByUser = false;
    this.wsPath = wsPath;
    this.attempt = 0;
    this.open();
  }

  private wsUrl(path: string): string {
    if (this.opts.baseUrl) return this.opts.baseUrl.replace(/^http/, "ws") + path;
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}${path}`;
  }

  private open(): void {
    if (!this.wsPath) return;
    this.opts.onStatus(this.attempt === 0 ? "connecting" : "reconnecting");
    const url = this.wsUrl(this.wsPath);
    const ws = this.opts.wsFactory ? this.opts.wsFactory(url) : new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.opts.onStatus("open");
    };
    ws.onmessage = (ev: MessageEvent) => {
      try {
        this.opts.onMessage(decodeServerMessage(String(ev.data)));
      } catch (err) {
        if (err instanceof ProtocolError) this.opts.onStatus("open", err.message);
        else throw err;
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.closedByUser) {
        this.opts.onStatus("closed");
        return;
      }
      const max = this.opts.maxAttempts ?? 8;
      if (this.attempt >= max) {
        this.opts.onStatus("closed", "gave up reconnecting");
        return;
      }
      const delay = backoffDelay(this.attempt);
      this.attempt += 1;
      this.opts.onStatus("reconnecting", `retry in ${delay} ms`);
      this.timer = setTimeout(() => this.open(), delay);
    };
  }

  send(cmd: ClientCommand): boolean {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(encodeCommand(cmd));
    return true;
  }

  close(): void {
    this.closedByUser = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    if (ws) ws.close();
  }
}
