import { describe, expect, it } from "vitest";
import { APP_NAME } from "../../src/version";

describe("app", () => {
  it("has a name", () => {
    expect(APP_NAME).toBe("ParkPilot");
  });
});
