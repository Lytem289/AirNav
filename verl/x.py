import pandas as pd

def create_small_dataset_fixed():
    print("正在注入 data_source 字段...")
    train_df = pd.read_parquet('./data/AirNav_rl.parquet')
    val_df = pd.read_parquet('./data/AirNav_rl_persona.parquet')
    
    small_train = train_df.sample(n=200, random_state=42)
    small_val = val_df.sample(n=20, random_state=42)

    def to_chat_format(prompt_text):
        clean_text = prompt_text.replace('<image>', '').replace('<|image_pad|>', '').strip()
        final_text = f"<image>\n{clean_text}"
        return [{'role': 'user', 'content': final_text}]
        
    def to_image_format(image_path_list):
        if isinstance(image_path_list, str):
            return [{"image": image_path_list}]
        elif hasattr(image_path_list, '__iter__'):
            formatted_list = []
            for path in image_path_list:
                formatted_list.append({"image": path})
            return formatted_list
        return []

    def build_reward_model(row):
        truth = str(row.get('future_actions', ''))
        return {"ground_truth": truth}

    text_col = 'instruction' if 'instruction' in small_train.columns else 'prompt'
    small_train['prompt'] = small_train[text_col].apply(to_chat_format)
    small_val['prompt'] = small_val[text_col].apply(to_chat_format)
    
    img_col = 'cur_view' if 'cur_view' in small_train.columns else 'images'
    small_train['images'] = small_train[img_col].apply(to_image_format)
    small_val['images'] = small_val[img_col].apply(to_image_format)
    
    small_train['reward_model'] = small_train.apply(build_reward_model, axis=1)
    small_val['reward_model'] = small_val.apply(build_reward_model, axis=1)

    # ✨ 核心修复：添加数据来源列
    small_train['data_source'] = 'airnav'
    small_val['data_source'] = 'airnav'
    
    small_train.to_parquet('./data/AirNav_rl_small.parquet', index=False)
    small_val.to_parquet('./data/AirNav_rl_persona_small.parquet', index=False)
    print("✅ 数据集构建完成！包含 reward_model 和 data_source！")

if __name__ == "__main__":
    create_small_dataset_fixed()