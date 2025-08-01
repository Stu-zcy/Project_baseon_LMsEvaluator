// assets/js/utils/attackStore.js

// 初始化配置存储
export function initAttackConfigs() {
    if (!localStorage.getItem('attackConfigs')) {
        const sampleConfigs = getSampleConfigs();
        localStorage.setItem('attackConfigs', JSON.stringify(sampleConfigs));
        return sampleConfigs;
    }
    return JSON.parse(localStorage.getItem('attackConfigs'));
}

// 获取示例配置
function getSampleConfigs() {
    return [
        {
            id: 1,
            name: "对抗攻击-1",
            type: "AdvAttack",
            strategy: "TextFoolerJin2019",
            description: "针对NLP模型的文本对抗攻击，通过替换同义词和干扰字符欺骗模型",
            status: "active",
            defenderEnabled: false,
            sent: true,
            executed: true,
            createdAt: "2025-07-15 10:30",
            params: {
                attack_nums: 3,
                defender: null,
            }
        },
        // 其他示例配置...
    ];
}

// 获取所有配置
export function getAttackConfigs() {
    const configs = localStorage.getItem('attackConfigs');
    return configs ? JSON.parse(configs) : [];
}

// 保存所有配置
export function saveAttackConfigs(configs) {
    localStorage.setItem('attackConfigs', JSON.stringify(configs));
}

// 更新或添加配置
export function updateAttackConfig(config) {
    const configs = getAttackConfigs();
    const index = configs.findIndex(c => c.id === config.id);
    
    if (index !== -1) {
        configs[index] = config;
    } else {
        configs.push(config);
    }
    
    saveAttackConfigs(configs);
    return configs;
}

// 删除配置
export function deleteAttackConfig(configId) {
    let configs = getAttackConfigs();
    configs = configs.filter(c => c.id !== configId);
    saveAttackConfigs(configs);
    return configs;
}

// 设置全局配置
export function setGlobalConfig(config) {
    localStorage.setItem('globalConfig', JSON.stringify(config));
}

// 获取全局配置
export function getGlobalConfig() {
    const config = localStorage.getItem('globalConfig');
    return config ? JSON.parse(config) : {
        general: { random_seed: 42, use_gpu: true },
        LM_config: { model: "bert_base_uncased" },
        task_config: {
            task: "TaskForSingleSentenceClassification",
            dataset: "语法可接受性判断",
            normal_training: false,
            epochs: 3
        }
    };
}