import { Composition } from "remotion";
import { Demo } from "./Demo";
import { FPS, TOTAL } from "./scenes";

export const Root = () => (
  <>
    <Composition id="Demo16x9" component={Demo} width={1920} height={1080} fps={FPS} durationInFrames={TOTAL} />
  </>
);
