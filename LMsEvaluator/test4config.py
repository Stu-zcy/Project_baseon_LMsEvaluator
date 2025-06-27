import yaml
import os
import subprocess

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'MeaeQ_config.yaml')
STEAL_SCRIPT = os.path.join(os.path.dirname(__file__), 'attack', 'MeaeQ', 'steal', 'model_steal', 'original_steal.py')


def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_victim_ckpt_path(config):
    model_name = config['model_config']['victim_model']
    task_name = config['task_config']['task_name']
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, 'saved_model', f'{task_name}-{model_name}-victim-model.pkl'),
        os.path.join(base_dir, f'{task_name}-{model_name}-victim-model.pkl'),
        os.path.join(base_dir, 'saved_model', f'{model_name}-victim-model.pkl'),
        os.path.join(base_dir, f'{model_name}-victim-model.pkl'),
    ]
    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt
    return candidates[0]


def build_args_from_config(config):
    args = []
    victim_ckpt = get_victim_ckpt_path(config)
    args += ['--victim_model_version', config['model_config']['victim_model']]
    args += ['--victim_model_checkpoint', victim_ckpt]
    args += ['--steal_model_version', config['model_config']['steal_model']]
    args += ['--steal_model_checkpoint', f"{config['model_config']['steal_model']}-steal-model.pkl"]
    args += ['--task_name', config['task_config']['task_name']]
    args += ['--num_labels', str(config['task_config']['num_labels'])]
    args += ['--tokenize_max_length', str(config['task_config']['tokenize_max_length'])]
    attack = config['attack_config']
    args += ['--method', attack['method']]
    args += ['--query_num', str(attack['query_num'])]
    args += ['--run_seed', str(attack['run_seed_arr'][0])]
    args += ['--pool_data_type', attack['pool_data_type']]
    args += ['--pool_data_source', attack['pool_data_source']]
    args += ['--pool_subsize', str(attack['pool_subsize'])]
    args += ['--prompt', attack['prompt']]
    args += ['--epsilon', str(attack['epsilon'])]
    args += ['--initial_sample_method', attack['initial_sample_method']]
    args += ['--initial_drk_model', attack['initial_drk_model']]
    args += ['--al_sample_batch_num', str(attack['al_sample_batch_num'])]
    args += ['--al_sample_method', str(attack['al_sample_method'])]
    train = config['train_config']
    args += ['--batch_size', str(train['batch_size'])]
    args += ['--optimizer', train['optimizer']]
    args += ['--learning_rate', str(train['learning_rate'])]
    args += ['--weight_decay', str(train['weight_decay'])]
    args += ['--num_epochs', str(train['num_epochs'])]
    args += ['--weighted_cross_entropy', str(train['weighted_cross_entropy'])]
    args += ['--visible_device', '0']
    return args


def main():
    config = load_config()
    args = build_args_from_config(config)
    print(args)
    cmd = ['python', STEAL_SCRIPT] + args
    print('[信息] 测试命令:')
    print(' '.join([str(x) for x in cmd]))
    print('\n[攻击主流程输出]:\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        print('[stderr]:\n', result.stderr)
    if result.returncode != 0:
        print('[错误] 攻击主流程执行失败！')
    else:
        print('[信息] 攻击主流程执行成功。')


if __name__ == '__main__':
    main()
