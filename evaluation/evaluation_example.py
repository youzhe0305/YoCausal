# 主辦方的評估腳本，參賽者無法碰到這段程式碼
def evaluate_model(evaluator: CausalDiffusionEvaluator, causal_dataset):
    total_causal_score = 0
    
    for video_normal, video_reversed, prompt in causal_dataset:
        # 隨機或均勻取樣 timestep
        t = sample_timestep() 
        
        # 呼叫參賽者實作的 API 獲取 Loss
        loss_normal = evaluator.get_denoising_loss(video_normal, prompt, t)
        loss_reversed = evaluator.get_denoising_loss(video_reversed, prompt, t)
        
        # 如果模型具備因果理解，反常影片的 Denoising Loss 應該顯著高於正常影片
        if loss_reversed > loss_normal:
            total_causal_score += 1

    return total_causal_score / len(causal_dataset)