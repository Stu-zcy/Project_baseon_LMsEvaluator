// 本地存储键名
const STORAGE_KEY = 'attackConfigurations';
const GLOBAL_CONFIG_KEY = 'globalConfiguration';

/**
 * 初始化配置存储
 */
function initializeConfigStorage() {
    if (!localStorage.getItem(STORAGE_KEY)) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify([]));
    }
    
    if (!localStorage.getItem(GLOBAL_CONFIG_KEY)) {
        localStorage.setItem(GLOBAL_CONFIG_KEY, JSON.stringify({
            general: {
                random_seed: 42,
                use_gpu: true
            },
            LM_config: {
                model: "bert_base_uncased",
                local_model:true
            },
            task_config: {
                task: "TaskForSingleSentenceClassification",
                dataset: "语法可接受性判断",
                normal_training: false,
                epochs: 3
            }
        }));
    }
}

/**
 * 保存配置列表到本地存储
 * @param {Array} configList 配置列表
 */
function saveConfigList(configList) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(configList));
}

/**
 * 从本地存储加载配置列表
 * @returns {Array} 配置列表
 */
function loadConfigList() {
    const storedConfigs = localStorage.getItem(STORAGE_KEY);
    return storedConfigs ? JSON.parse(storedConfigs) : [];
}

/**
 * 保存全局配置到本地存储
 * @param {Object} globalConfig 全局配置对象
 */
function saveGlobalConfig(globalConfig) {
    localStorage.setItem(GLOBAL_CONFIG_KEY, JSON.stringify(globalConfig));
}

/**
 * 从本地存储加载全局配置
 * @returns {Object} 全局配置对象
 */
function loadGlobalConfig() {
    const storedConfig = localStorage.getItem(GLOBAL_CONFIG_KEY);
    return storedConfig ? JSON.parse(storedConfig) : null;
}

// 导出函数供其他模块使用
export {
    initializeConfigStorage,
    saveConfigList,
    loadConfigList,
    saveGlobalConfig,
    loadGlobalConfig
};