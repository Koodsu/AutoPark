import { describe, expect, it } from "vitest";
import { gaugeLevel } from "../../src/hmi/ClearanceGauges";

describe("hmi helpers", () => {
  it("colours clearance gauges by distance", () => {
    expect(gaugeLevel(0.1)).toBe("danger");
    expect(gaugeLevel(0.5)).toBe("warn");
    expect(gaugeLevel(2)).toBe("ok");
  });
});
