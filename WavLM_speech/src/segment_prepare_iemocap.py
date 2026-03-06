import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

ROOT = "raw_IEMOCAP"
OUT_CSV = "data/iemocap_segments_full.csv"  # 注意：文件名改成了 full

def parse_anvil(anvil_path):
    """
    解析 anvil 文件，提取 (start, end, A, V, D)
    """
    try:
        tree = ET.parse(anvil_path)
    except Exception:
        return []
        
    root = tree.getroot()
    
    # 1. 寻找 Primary (Emotion/Segment) 和 Primitive (AVD) 轨道
    primary_track = None
    primitive_track = None

    for track in root.findall("./body/track"):
        name = track.get("name", "")
        if name.endswith(".Emotion"):
            primary_track = track
        if name.endswith(".Primitives"):
            primitive_track = track

    if primary_track is None or primitive_track is None:
        return []

    # 2. 建立索引映射
    prim_map = {} # index -> {act, val, dom}
    for el in primitive_track.findall("el"):
        idx = el.get("index")
        vals = {}
        for attr in el.findall("attribute"):
            name = attr.get("name", "").lower()
            text = attr.text.strip() if attr.text else ""
            try:
                # 解析 "act 4" -> 4
                val = int(text.split()[-1])
                if "activation" in name: vals['a'] = val
                elif "valence" in name: vals['v'] = val
                elif "dominance" in name: vals['d'] = val
            except:
                pass
        
        if 'a' in vals and 'v' in vals and 'd' in vals:
            prim_map[idx] = vals

    # 3. 匹配区间
    segments = []
    for el in primary_track.findall("el"):
        idx = el.get("index")
        if idx in prim_map:
            try:
                start = float(el.get("start"))
                end = float(el.get("end"))
                p = prim_map[idx]
                segments.append((start, end, p['a'], p['v'], p['d']))
            except:
                continue
                
    return segments

def find_wav_path(root, session, improv_id):
    """
    智能查找 WAV 文件
    优先级 1: dialog/wav/ (这是你刚才发现的路径)
    优先级 2: dialog/avi/DivX/ (Session 1 的备用路径)
    """
    # 路径 1: 标准音频路径
    p1 = os.path.join(root, session, "dialog/wav", f"{improv_id}.wav")
    if os.path.exists(p1): return p1
    
    # 路径 2: 视频提取路径 (备用)
    p2 = os.path.join(root, session, "dialog/avi/DivX", f"{improv_id}.wav")
    if os.path.exists(p2): return p2
    
    return None

def main():
    all_rows = []
    
    # 遍历 Session1 到 Session5
    sessions = [f"Session{i}" for i in range(1, 6)]
    
    print(f"🚀 开始全量扫描 Session 1-5 ...")

    for session in sessions:
        session_dir = os.path.join(ROOT, session)
        if not os.path.exists(session_dir):
            print(f"⚠️  跳过 {session} (目录不存在)")
            continue
            
        # 寻找 Attribute 文件夹
        # 这里要兼容两种可能的路径结构
        # 结构 A: dialog/EmoEvaluation/Attribute
        attr_dir = os.path.join(session_dir, "dialog/EmoEvaluation/Attribute")
        
        if not os.path.exists(attr_dir):
            print(f"⚠️  {session} 找不到 Attribute 目录，跳过")
            continue
            
        # 获取所有 anvil 文件
        files = [f for f in os.listdir(attr_dir) if f.endswith(".anvil")]
        print(f"👉 {session}: 发现 {len(files)} 个标注文件，正在解析...")
        
        count_success = 0
        for f in files:
            anvil_path = os.path.join(attr_dir, f)
            
            # 文件名示例: Ses01F_impro01_e3.anvil -> ID: Ses01F_impro01
            # 逻辑：取 "_e" 之前的所有部分
            if "_e" in f:
                improv_id = f.split("_e")[0]
            else:
                # 兼容其他命名情况
                improv_id = f.replace(".anvil", "")
            
            # 寻找音频
            wav_path = find_wav_path(ROOT, session, improv_id)
            
            if not wav_path:
                # 找不到音频，跳过
                continue
                
            # 解析数据
            segs = parse_anvil(anvil_path)
            
            if len(segs) > 0:
                count_success += 1
            
            for s, e, a, v, d in segs:
                all_rows.append({
                    "wav_path": wav_path, 
                    "start": s,
                    "end": e,
                    "activation": a,
                    "valence": v,
                    "dominance": d,
                    "session": session 
                })
        print(f"   -> 成功匹配并解析了 {count_success} 个对话")

    # 保存
    df = pd.DataFrame(all_rows)
    print(f"\n🎉 全量解析完成！共找到 {len(df)} 条数据")
    
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"已保存到: {OUT_CSV}")

if __name__ == "__main__":
    main()