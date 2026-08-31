declare module "gifenc" {
  export const GIFEncoder: () => {
    writeFrame: (
      index: Uint8Array,
      width: number,
      height: number,
      options: { palette: unknown; delay: number },
    ) => void;
    finish: () => void;
    bytesView: () => Uint8Array;
  };
  export const quantize: (rgba: Uint8Array, colors: number) => unknown;
  export const applyPalette: (rgba: Uint8Array, palette: unknown) => Uint8Array;
}
