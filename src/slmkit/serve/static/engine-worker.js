// Runs model.js in a Web Worker, so generating never freezes the page. The playground sends
// {id, type: "load", base, name} once, then {id, type: "generate", body} per request, and gets back
// {id, result} or {id, error}.
import { generate, loadModel } from "./model.js";

let loaded = null;
let name = "model";

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (type === "load") {
      name = data.name;
      loaded = await loadModel(data.base);
      self.postMessage({ id, result: { vocab: loaded.tokenizer.vocabSize } });
    } else if (type === "generate") {
      if (!loaded) throw new Error("the model has not loaded");
      self.postMessage({ id, result: generate(loaded, data.body, name) });
    }
  } catch (err) {
    self.postMessage({ id, error: String(err?.message ?? err) });
  }
};
