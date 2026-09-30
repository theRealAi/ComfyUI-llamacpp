import { app } from "/scripts/app.js";

const knownServers = new Map();

function serverKey(url, apiKey) {
    const base = String(url || "").trim().replace(/\/+$/, "");
    return `${base}\n${apiKey || ""}`;
}

async function fetchModels(url, apiKey) {
    const response = await fetch("/llamacpp/get_models", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            url,
            api_key: apiKey || "",
        }),
    });
    if (!response.ok) {
        throw new Error(response.statusText);
    }
    const models = await response.json();
    if (!Array.isArray(models)) {
        throw new Error("llama-server returned no model list");
    }
    return models;
}

app.registerExtension({
    name: "Comfy.LlamaCppNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LlamaCppConnectivity") {
            return;
        }

        const originalNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (originalNodeCreated) {
                originalNodeCreated.apply(this, arguments);
            }

            const node = this;
            const urlWidget = node.widgets.find((w) => w.name === "url");
            const modelWidget = node.widgets.find((w) => w.name === "model");
            const apiKeyWidget = node.widgets.find((w) => w.name === "api_key");
            const refreshButtonWidget = node.addWidget("button", "🔄 Reconnect");

            const connection = () => ({
                url: urlWidget.value,
                apiKey: apiKeyWidget ? apiKeyWidget.value : "",
            });

            const applyModels = (models, keepSelection) => {
                const prev = modelWidget.value;
                const values = Array.isArray(models) ? models.slice() : [];
                if (keepSelection && prev && !values.includes(prev)) {
                    values.unshift(prev);
                }
                modelWidget.options.values = values;
                if (prev && values.includes(prev)) {
                    if (modelWidget.value !== prev) {
                        modelWidget.value = prev;
                    }
                } else if (!keepSelection && values.length > 0 && modelWidget.value !== values[0]) {
                    modelWidget.value = values[0];
                }
                node.setDirtyCanvas(true);
            };

            const applyKnown = () => {
                const { url, apiKey } = connection();
                const cached = knownServers.get(serverKey(url, apiKey));
                if (cached) {
                    applyModels(cached, true);
                }
            };

            const refresh = async (notify) => {
                const { url, apiKey } = connection();
                const key = serverKey(url, apiKey);
                if (notify) {
                    refreshButtonWidget.name = "⏳ Fetching...";
                    node.setDirtyCanvas(true);
                }
                try {
                    const models = await fetchModels(url, apiKey);
                    knownServers.set(key, models);
                    applyModels(models, !notify);
                } catch (error) {
                    const cached = knownServers.get(key);
                    if (cached) {
                        applyModels(cached, true);
                    }
                    if (notify) {
                        console.error("Error fetching llama-server models:", error);
                        app.extensionManager.toast.add({
                            severity: "error",
                            summary: "llama-server connection error",
                            detail: "Make sure llama-server is running on the URL",
                            life: 5000,
                        });
                    }
                }
                if (notify) {
                    refreshButtonWidget.name = "🔄 Reconnect";
                    node.setDirtyCanvas(true);
                }
            };

            const onEndpointEdited = () => {
                applyKnown();
                refresh(false);
            };
            urlWidget.callback = onEndpointEdited;
            if (apiKeyWidget) {
                apiKeyWidget.callback = onEndpointEdited;
            }
            refreshButtonWidget.callback = () => refresh(true);
            node._llamacppApplyKnown = applyKnown;

            const previousConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                if (previousConfigure) {
                    previousConfigure.apply(this, arguments);
                }
                node._llamacppConfigured = true;
                const saved = info && info.widgets_values;
                const savedModel = Array.isArray(saved) ? saved[1] : "";
                if (savedModel && !modelWidget.value) {
                    const values = (modelWidget.options.values || []).slice();
                    if (!values.includes(savedModel)) {
                        values.push(savedModel);
                        modelWidget.options.values = values;
                    }
                    modelWidget.value = savedModel;
                }
                applyKnown();
                refresh(false);
            };

            setTimeout(() => {
                if (node._llamacppConfigured) {
                    return;
                }
                applyKnown();
                refresh(false);
            }, 0);
        };
    },

    loadedGraphNode(node) {
        if (node && node._llamacppApplyKnown) {
            node._llamacppApplyKnown();
        }
    },
});
