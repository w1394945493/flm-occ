import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist



def pre_recall_f1score_iou(
    gt,
    pd,
    list_precision,
    list_recall,
    list_f1score,
    list_iou,
):
    if gt.sum() > 0:
        tp = ( pd &  gt).sum().item()
        fp = ( pd & ~gt).sum().item()
        fn = (~pd &  gt).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
        f1score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.
    else:
        precision = 0.
        recall = 0.
        f1score = 0.
        iou = 0.
    list_precision.append(precision)
    list_recall.append(recall)
    list_f1score.append(f1score)
    list_iou.append(iou)


def stat_pre_recall_f1score_iou(
    list_pre,
    list_rec,
    list_f1s,
    list_iou,
    list_valid_query,
    list_occ_in_loss,
    hist_bins=5,
):
    if not isinstance(list_pre, np.ndarray):
        list_pre = np.array(list_pre)
    if not isinstance(list_rec, np.ndarray):
        list_rec = np.array(list_rec)
    if not isinstance(list_f1s, np.ndarray):
        list_f1s = np.array(list_f1s)
    if not isinstance(list_iou, np.ndarray):
        list_iou = np.array(list_iou)
    if not isinstance(list_valid_query, np.ndarray):
        list_valid_query = np.array(list_valid_query)
    if not isinstance(list_occ_in_loss, np.ndarray):
        list_occ_in_loss = np.array(list_occ_in_loss)
        
    mean_pre = list_pre.mean()
    mean_rec = list_rec.mean()
    mean_f1s = list_f1s.mean()
    mean_iou = list_iou.mean()
    mean_valid_query = list_valid_query.mean()
    mean_occ_in_loss = list_occ_in_loss.mean()
    
    median_pre = np.median(list_pre)
    median_rec = np.median(list_rec)
    median_f1s = np.median(list_f1s)
    median_iou = np.median(list_iou)
    median_valid_query = np.median(list_valid_query)
    median_occ_in_loss = np.median(list_occ_in_loss)
    
    hist_pre = np.histogram(list_pre, bins=hist_bins, range=(0, 1))[0] / len(list_pre)
    hist_rec = np.histogram(list_rec, bins=hist_bins, range=(0, 1))[0] / len(list_pre)
    hist_f1s = np.histogram(list_f1s, bins=hist_bins, range=(0, 1))[0] / len(list_pre)
    hist_iou = np.histogram(list_iou, bins=hist_bins, range=(0, 1))[0] / len(list_pre)
    hist_valid_query = np.histogram(list_valid_query, bins=hist_bins, range=(0, 1))[0] / len(list_pre)
    hist_occ_in_loss = np.histogram(list_occ_in_loss, bins=hist_bins, range=(0, 1))[0] / len(list_pre)
    
    str_hist_pre = ', '.join([f'{v:.2f}' for v in hist_pre])
    str_hist_rec = ', '.join([f'{v:.2f}' for v in hist_rec])
    str_hist_f1s = ', '.join([f'{v:.2f}' for v in hist_f1s])
    str_hist_iou = ', '.join([f'{v:.2f}' for v in hist_iou])
    str_hist_valid_query = ', '.join([f'{v:.2f}' for v in hist_valid_query])
    str_hist_occ_in_loss = ', '.join([f'{v:.2f}' for v in hist_occ_in_loss])
    
    # 5 colors
    list_color_emoji = [
        '⬛',  # 0.0 black
        '🟦',  # 0.2 blue
        '🟩',  # 0.4 green
        '🟨',  # 0.6 yellow
        '🟧',  # 0.8 orange
        '🟥',  # 1.0 red
    ]
    def get_color(value):
        return list_color_emoji[int(value//0.2)+1 if value > 1e-16 else 0]
    color_emoji_hist_pre = ' '.join([get_color(v) for v in hist_pre])
    color_emoji_hist_rec = ' '.join([get_color(v) for v in hist_rec])
    color_emoji_hist_f1s = ' '.join([get_color(v) for v in hist_f1s])
    color_emoji_hist_iou = ' '.join([get_color(v) for v in hist_iou])
    color_emoji_hist_valid_query = ' '.join([get_color(v) for v in hist_valid_query])
    color_emoji_hist_occ_in_loss = ' '.join([get_color(v) for v in hist_occ_in_loss])
    
    str_pre = f'Precision:   {mean_pre:.2f} | {median_pre:.2f} | {str_hist_pre} | {color_emoji_hist_pre}'
    str_rec = f'Recall:      {mean_rec:.2f} | {median_rec:.2f} | {str_hist_rec} | {color_emoji_hist_rec}'
    str_f1s = f'F1 Score:    {mean_f1s:.2f} | {median_f1s:.2f} | {str_hist_f1s} | {color_emoji_hist_f1s}'
    str_iou = f'IoU:         {mean_iou:.2f} | {median_iou:.2f} | {str_hist_iou} | {color_emoji_hist_iou}'
    str_valid_query = f'Valid Query: {mean_valid_query:.2f} | {median_valid_query:.2f} | {str_hist_valid_query} | {color_emoji_hist_valid_query}'
    str_occ_in_loss = f'Occ in Loss: {mean_occ_in_loss:.2f} | {median_occ_in_loss:.2f} | {str_hist_occ_in_loss} | {color_emoji_hist_occ_in_loss}'
    
    dict_stat = {
        'mean_pre': mean_pre,
        'mean_rec': mean_rec,
        'mean_f1s': mean_f1s,
        'mean_iou': mean_iou,
        'mean_valid_query': mean_valid_query,
        'mean_occ_in_loss': mean_occ_in_loss,
        'median_pre': median_pre,
        'median_rec': median_rec,
        'median_f1s': median_f1s,
        'median_iou': median_iou,
        'median_valid_query': median_valid_query,
        'median_occ_in_loss': median_occ_in_loss,
        'hist_pre': hist_pre,
        'hist_rec': hist_rec,
        'hist_f1s': hist_f1s,
        'hist_iou': hist_iou,
        'hist_valid_query': hist_valid_query,
        'hist_occ_in_loss': hist_occ_in_loss,
        'str_pre': str_pre,
        'str_rec': str_rec,
        'str_f1s': str_f1s,
        'str_iou': str_iou,
        'str_valid_query': str_valid_query,
        'str_occ_in_loss': str_occ_in_loss,
        'list_pre': list_pre,
        'list_rec': list_rec,
        'list_f1s': list_f1s,
        'list_iou': list_iou,
        'list_valid_query': list_valid_query,
        'list_occ_in_loss': list_occ_in_loss,
    }
    
    return dict_stat


class MeanIoUAcc:
    def __init__(self, num_classes, ignore_index=-1, device='cuda'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)
    
    def update(self, pd, gt):
        """
        pd: (N,) long tensor
        gt: (N,) long tensor
        """
        mask = (gt != self.ignore_index)
        pd = pd[mask]
        gt = gt[mask]
        
        if pd.numel() == 0:
            return
        
        cm = torch.bincount(
            self.num_classes * gt + pd,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm
    
    def sync(self):
        if dist.is_initialized():
            dist.all_reduce(self.confusion_matrix)
            
        return self
    
    def compute(self):
        """
        Returns:
            mean_iou: float
        """

        tp = torch.diag(self.confusion_matrix).float()
        fp = self.confusion_matrix.sum(dim=0).float() - tp
        fn = self.confusion_matrix.sum(dim=1).float() - tp
        
        denom = tp + fp + fn
        ious = tp / denom
        ious[denom == 0] = float('nan')  # if no ground truth, do not include in evaluation
        
        mean_iou = torch.nanmean(ious).item()
        
        return mean_iou, ious

def generate_eval_tables(ious, namemap, iou_bin=None):
    """
    输入 IoU 数据，返回 Markdown 和 LaTeX 表格字符串。
    
    参数：
        ious: Tensor，形状 [num_classes]，包含每个类别的 IoU
        namemap: list 或 dict，namemap[i] 是类别 i 的名称
        iou_bin: （可选）Tensor，若提供则使用 iou_bin[1].item() 作为总 IoU；
                 若不提供，则用 ious[1:].nanmean().item() 作为总 IoU
    
    返回：
        markdown_table: str, Markdown 格式的横向表格
        latex_table: str, LaTeX 格式的表格
    """

    def escape_markdown(text):
        return str(text).replace('|', '\\|').replace('\n', ' ')

    def escape_latex(text):
        s = str(text)
        for f, r in [('%', '\\%'), ('_', '\\_'), ('&', '\\&'), ('#', '\\#'), ('$','\\$'), ('{','\\{'), ('}','\\}')]:
            s = s.replace(f, r)
        return s

    # 1. 确定 IoU 和 mIoU 值
    overall_iou = iou_bin[1].item() if iou_bin is not None else ious[1:].nanmean().item()
    miou = ious[1:].nanmean().item()

    # 2. 构造表头和数值（跳过第0类）
    headers = ["Method", "IoU", "mIoU"]
    values = [
        f"{overall_iou * 100:.2f}",
        f"{miou * 100:.2f}"
    ]

    # 添加每个类别的名称和数值
    for i, iou_i in enumerate(ious):
        if i > 0:  # 跳过第0类（背景）
            class_name = namemap[i] if isinstance(namemap, (list, tuple)) else namemap.get(i, f"Class{i}")
            headers.append(escape_markdown(class_name))
            values.append(f"{iou_i.item() * 100:.2f}")
    
    # 3. 生成 Markdown 表格
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["-----"] * len(headers)) + " |",
        "|  Ours | " + " | ".join(values) + " |",
    ]
    markdown_table = "\n".join(markdown_rows)

    # 4. 生成 LaTeX 表格
    latex_headers = [escape_latex(h) for h in headers]
    latex_values = [v.replace('%', '\\%') for v in values]  # 在 LaTeX 中 % 需要转义

    latex_table = (
        # "\\begin{table}[h]\n"
        # "\\centering\n"
        # "\\begin{tabular}{|" + "c|" * len(headers) + "}\n"
        # "\\hline\n" +
        " & ".join(latex_headers) + " \\\\\n\\hline\n" +
        " & ".join(latex_values) + " \\\\\n\\hline\n"
        # "\\end{tabular}\n"
        # "\\caption{Evaluation Metrics (IoU in \\%)}\n"
        # "\\end{table}"
    )

    _str = ''.join([
        f'| IoU: {iou_bin[1].item()*100:.2f}\n',
        f'| mIoU: {ious[1:].nanmean().item()*100:.2f}\n'
    ] + [
        f'| {namemap[i]}: {iou_i.item()*100:.2f}\n'
        for i, iou_i in enumerate(ious) if i > 0
    ])

    return _str, markdown_table, latex_table

