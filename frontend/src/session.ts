import { RunClient } from "./net/client";
import type { ClientCommand } from "./net/protocol";
import { appStore } from "./store/store";

/** The single connection between the HMI and the backend. */
export const client = new RunClient({
  onMessage: (msg) => appStore.getState().applyMessage(msg),
  onStatus: (status, detail) => appStore.getState().setStatus(status, detail),
});

export function send(cmd: ClientCommand): void {
  if (!client.send(cmd)) appStore.setState({ lastAck: "not connected" });
}
