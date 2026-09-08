import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.LlamaCppNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LlamaCppConnectivity") {
            return;
        }

        const originalNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            if (originalNodeCreated) {
                originalNodeCreated.apply(this, arguments);
            }

            const urlWidget = this.widgets.find((w) => w.name === "url");
            const modelWidget = this.widgets.find((w) => w.name === "model");
            const apiKeyWidget = this.widgets.find((w) => w.name === "api_key");
            const refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");

            const fetchModels = async (url, apiKey) => {
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

                if (response.ok) {
                    const models = await response.json();
                    return models;
                }
                throw new Error(response.statusText);
            };

            const updateModels = async () => {
                refreshButtonWidget.name = "⏳ Fetching...";
                this.setDirtyCanvas(true);

                let models = [];
                try {
                    models = await fetchModels(urlWidget.value, apiKeyWidget ? apiKeyWidget.value : "");
                } catch (error) {
                    console.error("Error fetching llama-server models:", error);
                    app.extensionManager.toast.add({
                        severity: "error",
                        summary: "llama-server connection error",
                        detail: "Make sure llama-server is running on the URL",
                        life: 5000,
                    });
                    refreshButtonWidget.name = "🔄 Reconnect";
                    this.setDirtyCanvas(true);
                    return;
                }

                const prevValue = modelWidget.value;
                modelWidget.options.values = models;
                if (models.includes(prevValue)) {
                    modelWidget.value = prevValue;
                } else if (models.length > 0) {
                    modelWidget.value = models[0];
                }

                refreshButtonWidget.name = "🔄 Reconnect";
                this.setDirtyCanvas(true);
            };

            urlWidget.callback = updateModels;
            refreshButtonWidget.callback = updateModels;
            if (apiKeyWidget) {
                apiKeyWidget.callback = updateModels;
            }

            await updateModels();
        };
    },
});
