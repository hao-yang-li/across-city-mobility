import os
import time
import yaml
import json
import csv
import numpy as np
import random
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import sys
import re
from utils import LLM
from constant import PLATFORM

VERSION = "v1"
CONFIG_PATH = "config.yaml"

EDUCATION_LEVEL_DESCRIPTIONS = {
    "Lowest": "at the lowest level in Virginia",
    "Low": "at a low level in Virginia",
    "Medium": "at a medium level in Virginia",
    "High": "at a high level in Virginia",
    "Highest": "at the highest level in Virginia"
}

CBG_SUMMARY_UNAVAILABLE = "No known POIs available for preview in this CBG."

EARTH_RADIUS_M = 6371000

def load_config(config_path):
    # 显式指定 encoding='utf-8' 来避免 Windows gbk 解码错误
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_agent_profiles(profile_path):
    with open(profile_path, 'r') as file:
        return json.load(file)

# --- CBG to City ---
def load_cbg_city_mapping(mapping_path):
    """
    加载 CBG Code 到 City Name 的映射
    """
    cbg_to_city = {}
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cbg_code = str(row['CBG Code']).strip()
                city_name = row['City Name'].strip()
                cbg_to_city[cbg_code] = city_name
        logging.debug(f"Loaded CBG to City mapping with {len(cbg_to_city)} entries.")
    except Exception as e:
        logging.error(f"Failed to load CBG to City mapping from {mapping_path}: {e}")
        raise
    return cbg_to_city

# --- 加载 CBG 数据 ---
def load_cbg_data(stats_path, summary_path, poverty_path, education_path, cities, year_filter):
    """
    加载并处理 CBG 人口、坐标、摘要、收入和教育数据。
    """
    cbg_data = {}
    cbg_summary = {}
    cbg_income = {}
    cbg_education = {}
    #加载人口和坐标
    with open(stats_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['Year']) == year_filter and row['City Name'] in cities:
                cbg_code = str(row['CBG Code'])
                try:
                    population = int(row['Population'])
                except ValueError:
                    print(f"警告: CBG {cbg_code} 的人口数据无效，跳过。")
                    continue
                try:
                    centroid_str = row['Centroid']
                    coords_str = centroid_str.replace('POINT (', '').replace(')', '')
                    lon, lat = map(float, coords_str.split())
                    centroid = np.array([lat, lon])
                except Exception as e:
                    print(f"警告: CBG {cbg_code} 的 Centroid 数据格式错误，跳过。错误: {e}")
                    continue
                cbg_data[cbg_code] = {
                    'population': population,
                    'centroid': centroid # 以 [lat, lon] 形式存储，后续计算转为米
                }
    # 加载CBG摘要
    with open(summary_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        cbg_code_idx = 0
        poi_categories = headers[1:]
        for row in reader:
            cbg_code = str(row[cbg_code_idx])
            if cbg_code in cbg_data:
                poi_counts = {}
                for category, count_str in zip(poi_categories, row[1:]):
                    if count_str and count_str != '0':
                        try:
                            count = int(count_str)
                            poi_counts[category] = count
                        except ValueError:
                            continue
                if not poi_counts:
                    summary_text = CBG_SUMMARY_UNAVAILABLE
                else:
                    sorted_poi = sorted(poi_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    summary_parts = [f"{category} ({count})" for category, count in sorted_poi]
                    summary_text = f"Top POI categories: {'; '.join(summary_parts)}."
                cbg_summary[cbg_code] = summary_text
    # 加载收入数据
    try:
        with open(poverty_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get('Year', 0)) == 2019:
                    cbg_code = str(row['CBG Code']).strip()
                    if cbg_code in cbg_data:
                        try:
                            income = int(row['Median Household Income'])
                            cbg_income[cbg_code] = income
                        except (ValueError, KeyError):
                            cbg_income[cbg_code] = "Unknown Income"
    except Exception as e:
       logging.error(f"Failed to load income data from {poverty_path}: {e}")
    # 加载教育数据
    try:
        with open(education_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cbg_code = str(row.get('CBG', row.get('CBG Code', ''))).strip()
                if cbg_code and cbg_code in cbg_data:
                    raw_education_level = row.get('Education_Level', 'Unknown').strip()
                    education_desc = EDUCATION_LEVEL_DESCRIPTIONS.get(raw_education_level, f"Education Level: {raw_education_level}")
                    cbg_education[cbg_code] = education_desc
    except Exception as e:
       logging.error(f"Failed to load education data from {education_path}: {e}")
    # 收入和教育数据合并到 cbg_data 字典中
    for cbg_code in cbg_data.keys():
        cbg_data[cbg_code]['income'] = cbg_income.get(cbg_code, "Income Data Unavailable")
        cbg_data[cbg_code]['education'] = cbg_education.get(cbg_code, "Education Data Unavailable")
    return cbg_data, cbg_summary

# --- 计算两个 CBG 之间的距离 ---
def calculate_distance(cbg1_centroid, cbg2_centroid):
    """
    计算两个 CBG 中心点之间的近似欧氏距离 (米)
    输入是 np.array([lat, lon]) 度
    """
    # 将度转换为弧度
    lat1_rad = np.radians(cbg1_centroid[0])
    lon1_rad = np.radians(cbg1_centroid[1])
    lat2_rad = np.radians(cbg2_centroid[0])
    lon2_rad = np.radians(cbg2_centroid[1])

    # 计算经纬度差值 (弧度)
    delta_lat_rad = lat2_rad - lat1_rad
    delta_lon_rad = lon2_rad - lon1_rad

    # 近似欧氏距离 (弧度)
    euclidean_delta_rad = np.sqrt(delta_lat_rad**2 + delta_lon_rad**2)

    # 转换为米 (使用地球半径)
    distance_meters = EARTH_RADIUS_M * euclidean_delta_rad

    return distance_meters

# --- d-EPR 模型 ---
def sample_waiting_time(beta=0.8, tau=17):
    """
    根据 d-EPR 模型的等待时间分布 P(Δt) ∝ Δt^{-1 - β} exp(-Δt / τ) 采样
    注意：在基于时间步的模拟中，这个函数可能直接返回 1 步，
    或者用于计算下一次移动的时间点。这里简化处理。
    """
    # 简化为在每个时间步都进行决策
    return 1

def calculate_p_new(S, rho=0.6, gamma=0.21):
    """计算探索新地点的概率 P_new = rho * S^(-gamma)"""
    if S == 0:
        # 如果没有访问过任何地方，强制探索
        return 1.0
    try:
        return rho * (S ** (-gamma))
    except OverflowError:
        # 处理 S 非常大导致的计算溢出
        return 0.0

def get_top_candidates(current_cbg_code, cbg_data, cbg_summary, cbg_to_city, top_n=20):
    """
    根据 d-EPR 引力模型 (pop_i * pop_j / dist^2) 获取前 N=20 个候选 CBG。
    只返回人口和距离信息，不返回计算出的 score
    同时包含 City 信息
    """
    candidates = []
    current_data = cbg_data.get(current_cbg_code)
    if not current_data:
        return candidates # 如果当前 CBG 无效，返回空列表
    current_pop = current_data['population']
    current_centroid = current_data['centroid']
    scores = []
    for candidate_cbg_code, candidate_data in cbg_data.items():
        if candidate_cbg_code == current_cbg_code:
            continue # 排除当前位置
        candidate_pop = candidate_data['population']
        candidate_centroid = candidate_data['centroid']
        distance_m = calculate_distance(current_centroid, candidate_centroid)
        if distance_m == 0:
            continue # 避免除以零
        # 计算得分 (pop_i * pop_j) / distance^2 用于排序，不包含在返回结果中
        score = (current_pop * candidate_pop) / (distance_m ** 2)
        scores.append((
            candidate_cbg_code,
            score,
            candidate_pop,
            distance_m, # 单位：米
            candidate_data.get('income', 'Income Data Unavailable'), # 新增 Income
            candidate_data.get('education', 'Education Data Unavailable') # 新增 Education
        ))
    # 按得分排序并取前 N 个
    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:top_n]
    for cbg_code, score, pop, dist_m, income, education in top_scores:
        summary = cbg_summary.get(cbg_code, CBG_SUMMARY_UNAVAILABLE)
        city = cbg_to_city.get(cbg_code, "Unknown City")
        candidates.append({
            "CBG_Code": cbg_code,
            "Population": pop,
            "Distance": dist_m,
            "Summary": summary,
            "City": city,
            "Income": income,
            "Education": education
        })
    return candidates

# --- Agent 类 ---
class Agent:
    def __init__(self, agent_id, profile, initial_cbg, cbg_data, cbg_summary, cbg_to_city, llm_client, traj_dir):
        self.id = agent_id
        self.profile = profile
        self.city = profile.get("City", "Unknown City")
        self.income = profile.get("Income", "Unknown")
        raw_higher_edu = profile.get("Higher_Edu", "Unknown")
        self.higher_edu = EDUCATION_LEVEL_DESCRIPTIONS.get(raw_higher_edu, f"Education Level: {raw_higher_edu}")
        # ---------------------------------------------------
        self.current_cbg = initial_cbg
        self.cbg_data = cbg_data
        self.cbg_summary = cbg_summary
        self.cbg_to_city = cbg_to_city
        self.llm_client = llm_client
        self.visited_cbgs = defaultdict(int)
        self.traj_dir = traj_dir
        self.visited_cbgs[self.current_cbg] += 1
        self._initialize_trajectory_file()

    def _initialize_trajectory_file(self):
        traj_file_path = os.path.join(self.traj_dir, f"R_{self.id}.json")
        if not os.path.exists(traj_file_path): # 仅在文件不存在时初始化
            initial_file_data = {
                "agent_profile": self.profile,
                "initial_cbg": self.current_cbg,
                "migration_history": []
            }
            try:
                with open(traj_file_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_file_data, f, indent=2, ensure_ascii=False)
                # logging.info(f"Created and initialized trajectory file for Agent {self.id} at {traj_file_path}")
                logging.debug(f"Created and initialized trajectory file for Agent {self.id} at {traj_file_path}")
            except Exception as e:
                logging.error(f"Failed to create/initialize trajectory file for Agent {self.id} at {traj_file_path}: {e}")

    def _log_decision(self, step, S, P_new, action, from_cbg, to_cbg, candidates, chosen_cbg_code, reasoning):
        """记录决策过程到轨迹文件中 ，实时写入"""
        # 获取当前 CBG 的摘要 (迁移前)
        from_summary = self.cbg_summary.get(from_cbg, CBG_SUMMARY_UNAVAILABLE)
        from_city = self.cbg_to_city.get(from_cbg, "Unknown City") # 新增 City
        # 获取目标 CBG 的摘要 (迁移后)
        to_summary = self.cbg_summary.get(to_cbg, CBG_SUMMARY_UNAVAILABLE)
        to_city = self.cbg_to_city.get(to_cbg, "Unknown City") # 新增 City
        from_data = self.cbg_data.get(from_cbg, {})
        to_data = self.cbg_data.get(to_cbg, {})
        from_income = from_data.get('income', 'Income Data Unavailable')
        from_education = from_data.get('education', 'Education Data Unavailable')
        to_income = to_data.get('income', 'Income Data Unavailable')
        to_education = to_data.get('education', 'Education Data Unavailable')
        # 准备要写入的决策记录
        decision_record = {
            "step": step,
            "S": S,
            "P_new": P_new,
            "action": action, # 'explore' or 'return'
            "from_cbg": from_cbg,
            "from_city": from_city,
            "from_summary": from_summary,
            "from_income": from_income,
            "from_education": from_education,
            "to_cbg": to_cbg,
            "to_city": to_city,
            "to_summary": to_summary,
            "to_income": to_income,
            "to_education": to_education,
            "candidates": candidates, # 包含 CBG_Code, Population, Distance (米), Summary, City, Income, Education
            "chosen_cbg": chosen_cbg_code,
            "llm_thinking": reasoning
        }
        traj_file_path = os.path.join(self.traj_dir, f"R_{self.id}.json")
        try:
            with open(traj_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if "migration_history" not in existing_data:
                existing_data["migration_history"] = []
            if not any(record.get("step") == step for record in existing_data["migration_history"]):
                existing_data["migration_history"].append(decision_record)
            else:
                logging.warning(f"Step {step} record already exists for Agent {self.id}. Skipping append.")
            with open(traj_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from trajectory file for Agent {self.id} at {traj_file_path}: {e}.")
        except Exception as e:
            logging.error(f"Failed to update trajectory file for Agent {self.id} at {traj_file_path}: {e}")

    def decide_next_location(self, step, max_retries=3):
        """使用 d-EPR 模型和 LLM 决策下一个地点"""
        # 记录迁移前的位置
        from_cbg = self.current_cbg
        # 1. 等待时间选择
        # delta_t = sample_waiting_time()
        # 2. 行动选择 (计算 P_new)
        S = len(self.visited_cbgs)
        P_new = calculate_p_new(S)
        # 3. 决策探索还是返回
        action = np.random.choice(['explore', 'return'], p=[P_new, 1 - P_new])
        if action == 'explore':
            # --- 探索 ---
            # 获取候选 CBG (不包含预计算的 Score, 包含 City, Income, Education)
            candidates = get_top_candidates(self.current_cbg, self.cbg_data, self.cbg_summary, self.cbg_to_city, top_n=20)
            if not candidates:
                # 记录决策日志 (无有效候选)
                # 传递空的 chosen_cbg_code 和 reasoning
                self._log_decision(step, S, P_new, 'explore', from_cbg, self.current_cbg, [], self.current_cbg, "No valid candidates.")
                return # Stay in current location
            # 获取当前 CBG 的人口、城市、收入和教育
            current_data = self.cbg_data.get(self.current_cbg, {})
            current_pop = current_data.get('population', 'Unknown')
            current_city = self.cbg_to_city.get(self.current_cbg, "Unknown City")
            current_income = current_data.get('income', 'Income Data Unavailable')
            current_education = current_data.get('education', 'Education Data Unavailable')
            # 构建 Prompt 给 LLM
            current_summary = self.cbg_summary.get(self.current_cbg, CBG_SUMMARY_UNAVAILABLE)
            prompt = f"""
You are a resident of {self.city} making a decision about your next location to visit. Your personal profile and the principles of human mobility should guide your choice.

**Your Profile:**
- City of Residence: {self.city}
- Income: {self.income}
- Higher Education Level: {self.higher_edu}

**Current Location:**
- CBG Code: {self.current_cbg}
- City: {current_city}
- Population: {current_pop}
- Median Household Income: {current_income}
- Education Level: {current_education}
- Summary: {current_summary}

**Decision Context:**
When exploring, you tend to select a new location probabilistically based on the "gravitational" pull between your current location and potential destinations. This pull is stronger when:
1.  The population of your current location is large.
2.  The population of the candidate location is large.
3.  The distance between your current and candidate locations is short.
Specifically, the likelihood of choosing a candidate is proportional to (Current_Population * Candidate_Population) / (Distance^2).

Below is a list of top candidate CBGs based on this principle. Your selection should align with this principle and your profile.
**Candidate Locations for Exploration:**
"""
            for i, cand in enumerate(candidates):
                prompt += f"""
{i+1}. CBG Code: {cand['CBG_Code']}
   - City: {cand['City']}
   - Population: {cand['Population']}
   - Distance from current: {cand['Distance']:.0f} meters
   - Median Household Income: {cand['Income']}
   - Education Level: {cand['Education']}
   - Summary: {cand['Summary']}
"""
            prompt += f"""
**Instructions:**
Based on your profile (especially income and education) and the exploration principle described above, choose ONE CBG Code from the list above to move to.

**Response Format:**
Respond ONLY with a valid JSON object containing two keys:
1. "chosen_cbg_code": The CBG Code of your chosen destination (e.g., "51550001001").
2. "reasoning": A short sentence explaining your reasoning for choosing this CBG.

Example:
{{
  "chosen_cbg_code": "51550001001",
  "reasoning": "I chose this CBG because it has a high population and is relatively close, aligning with the d-EPR exploration principle."
}}
"""

            logging.info(f"[PROMPT_SENT_TO_LLM] Agent {self.id} (Explore) at Step {step}:\n{prompt}")
            response = None
            for attempt in range(max_retries + 1): # 0, 1, 2, 3 (共4次尝试，包括初始尝试)
                try:
                    response = self.llm_client.generate(prompt)
                    break
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1}/{max_retries + 1} failed for Agent {self.id} (Explore): {e}")
                    if attempt < max_retries:
                        logging.info(f"Retrying in 2 seconds...")
                        time.sleep(2) # 等待2秒后重试
                    else:
                        # 所有重试都失败了
                        logging.critical(f"Agent {self.id} failed to get LLM response for exploration after {max_retries + 1} attempts. Exiting simulation.")
                        sys.exit(1) # 退出整个程序
            logging.info(f"[LLM_RESPONSE_RECEIVED] Agent {self.id} (Explore) at Step {step}:\n{response}")
            chosen_cbg_code = self.current_cbg
            reasoning = "LLM response parsing failed or invalid choice."
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    temp_chosen_cbg = str(parsed_response.get("chosen_cbg_code", "")).strip()
                    temp_reasoning = parsed_response.get("reasoning", "No reasoning provided.")
                    # 验证选择的 CBG 是否有效
                    if temp_chosen_cbg in [c['CBG_Code'] for c in candidates]:
                        chosen_cbg_code = temp_chosen_cbg
                        reasoning = temp_reasoning
                    else:
                         logging.warning(f"Agent {self.id} LLM chose invalid CBG '{temp_chosen_cbg}' for exploration. Staying.")
                         reasoning = f"Invalid CBG chosen by LLM: {temp_chosen_cbg}. Reasoning: {temp_reasoning}"
                else:
                    logging.error(f"Agent {self.id} LLM response did not contain valid JSON for exploration.")
            except json.JSONDecodeError as je:
                logging.error(f"Agent {self.id} LLM response JSON decoding failed for exploration: {je}")
            except Exception as e:
                logging.error(f"Unexpected error parsing LLM response for Agent {self.id} (Explore): {e}")
            # 更新 Agent 状态并记录决策
            self.current_cbg = chosen_cbg_code
            self.visited_cbgs[chosen_cbg_code] += 1
            # 记录决策日志
            self._log_decision(step, S, P_new, 'explore', from_cbg, chosen_cbg_code, candidates, chosen_cbg_code, reasoning)
        else: # action == 'return'
            # --- 返回 ---
            if len(self.visited_cbgs) <= 1:
                 # 记录决策日志
                 # 传递空的 chosen_cbg_code 和 reasoning
                 self._log_decision(step, S, P_new, 'return', from_cbg, self.current_cbg, [], self.current_cbg, "Only one location visited, cannot return.")
                 return # Simplification: Stay
            # 构建返回候选列表 (已访问的 CBG，不包括当前位置)
            return_candidates = {cbg: count for cbg, count in self.visited_cbgs.items() if cbg != self.current_cbg}
            if not return_candidates:
                 # 记录决策日志-
                 self._log_decision(step, S, P_new, 'return', from_cbg, self.current_cbg, [], self.current_cbg, "No other visited locations to return to.")
                 # ---------------------------------------------------
                 return
            # 计算返回概率 (与访问次数成正比)
            cbgs = list(return_candidates.keys())
            counts = np.array(list(return_candidates.values()))
            probs = counts / counts.sum()
            # 构建 Prompt 给 LLM
            current_summary = self.cbg_summary.get(self.current_cbg, CBG_SUMMARY_UNAVAILABLE)
            current_city = self.cbg_to_city.get(self.current_cbg, "Unknown City")
            current_data = self.cbg_data.get(self.current_cbg, {})
            current_income = current_data.get('income', 'Income Data Unavailable')
            current_education = current_data.get('education', 'Education Data Unavailable')
            prompt = f"""
You are a resident of {self.city} making a decision about your next location to visit. Your personal profile and the principles of human mobility should guide your choice.

**Your Profile:**
- City of Residence: {self.city}
- Income: {self.income}
- Higher Education Level: {self.higher_edu} 

**Current Location:**
- CBG Code: {self.current_cbg}
- City: {current_city}
- Median Household Income: {current_income}
- Education Level: {current_education}
- Summary: {current_summary}

**Decision Context:**
When returning, you tend to select a previously visited location with a probability proportional to the number of times you have visited it (preference for familiar places).

**Previously Visited Locations (Candidates for Return):**
"""
            for cbg_code, count in return_candidates.items():
                summary = self.cbg_summary.get(cbg_code, CBG_SUMMARY_UNAVAILABLE)
                city = self.cbg_to_city.get(cbg_code, "Unknown City") # 新增 City
                visited_data = self.cbg_data.get(cbg_code, {})
                income = visited_data.get('income', 'Income Data Unavailable')
                education = visited_data.get('education', 'Education Data Unavailable')
                prompt += f"""
- CBG Code: {cbg_code}
  - City: {city}
  - Visit Count: {count}
  - Median Household Income: {income}
  - Education Level: {education}
  - Summary: {summary}
"""
            prompt += f"""
**Instructions:**
Based on your profile (especially income and education) and the return principle (preferring frequently visited places), choose ONE CBG Code from the list above to return to.

**Response Format:**
Respond ONLY with a valid JSON object containing two keys:
1. "chosen_cbg_code": The CBG Code of your chosen destination (e.g., "51550001002").
2. "reasoning": A short sentence explaining your reasoning for choosing this CBG.

Example:
{{
  "chosen_cbg_code": "51550001002",
  "reasoning": "I chose this CBG because I have visited it many times before, and the d-EPR return principle favors familiar places."
}}
"""

            logging.info(f"[PROMPT_SENT_TO_LLM] Agent {self.id} (Return) at Step {step}:\n{prompt}")
            response = None
            for attempt in range(max_retries + 1): # 0, 1, 2, 3
                try:
                    response = self.llm_client.generate(prompt)
                    break
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1}/{max_retries + 1} failed for Agent {self.id} (Return): {e}")
                    if attempt < max_retries:
                        logging.info(f"Retrying in 2 seconds...")
                        time.sleep(2) # 等待2秒后重试
                    else:
                        # 所有重试都失败了
                        logging.critical(f"Agent {self.id} failed to get LLM response for return after {max_retries + 1} attempts. Exiting simulation.")
                        sys.exit(1) # 退出整个程序
            logging.info(f"[LLM_RESPONSE_RECEIVED] Agent {self.id} (Return) at Step {step}:\n{response}")
            chosen_cbg_code = self.current_cbg
            reasoning = "LLM response parsing failed or invalid choice."
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    temp_chosen_cbg = str(parsed_response.get("chosen_cbg_code", "")).strip()
                    temp_reasoning = parsed_response.get("reasoning", "No reasoning provided.")
                    # 验证选择的 CBG 是否有效
                    if temp_chosen_cbg in cbgs:
                        chosen_cbg_code = temp_chosen_cbg
                        reasoning = temp_reasoning
                    else:
                         logging.warning(f"Agent {self.id} LLM chose invalid CBG '{temp_chosen_cbg}' for return. Staying.")
                         reasoning = f"Invalid CBG chosen by LLM: {temp_chosen_cbg}. Reasoning: {temp_reasoning}"
                else:
                    logging.error(f"Agent {self.id} LLM response did not contain valid JSON for return.")
            except json.JSONDecodeError as je:
                logging.error(f"Agent {self.id} LLM response JSON decoding failed for return: {je}")
            except Exception as e:
                logging.error(f"Unexpected error parsing LLM response for Agent {self.id} (Return): {e}")
            # 更新 Agent 状态并记录决策
            self.current_cbg = chosen_cbg_code
            self.visited_cbgs[chosen_cbg_code] += 1
            # 为返回的 candidates 准备数据 (包含 visit count, summary, city, income, education)
            return_candidates_with_summary = []
            for code, count in return_candidates.items():
                 summary = self.cbg_summary.get(code, CBG_SUMMARY_UNAVAILABLE)
                 city = self.cbg_to_city.get(code, "Unknown City")
                 visited_data = self.cbg_data.get(code, {})
                 income = visited_data.get('income', 'Income Data Unavailable')
                 education = visited_data.get('education', 'Education Data Unavailable')
                 return_candidates_with_summary.append({
                     "CBG_Code": code,
                     "Visit_Count": count,
                     "Summary": summary,
                     "City": city,
                     "Income": income,
                     "Education": education
                 })
            self._log_decision(step, S, P_new, 'return', from_cbg, chosen_cbg_code, return_candidates_with_summary, chosen_cbg_code, reasoning)

# --- 主函数 ---
def main():
    start_time = datetime.now()
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{VERSION}_{timestamp_str}"
    # 加载配置
    config = load_config(CONFIG_PATH)
    output_dir = os.path.join(config['output_base_dir'], run_dir)
    log_dir = os.path.join(output_dir, "logs")
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'simulation.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

    logging.info(f"Starting simulation {run_dir}")

    cbg_to_city = load_cbg_city_mapping(config['data_files']['cbg_city_lookup'])
    cbg_data, cbg_summary = load_cbg_data(
        config['data_files']['cbg_stats'],
        config['data_files']['cbg_summary'],
        config['data_files']['cbg_poverty'],
        config['data_files']['cbg_education'],
        config['cities'],
        config['year_filter']
    )
    llm_client = LLM(
        model_name=config['llm']['model_name'],
        platform=config['llm']['platform'],
        api_key=config['llm'].get('api_key')
    )
    agent_profiles = load_agent_profiles("agent_initialization/agent_profiles.json")
    agents = []
    for i, profile in enumerate(agent_profiles):
        initial_cbg_from_profile = str(profile.get('CBG'))
        if not initial_cbg_from_profile or initial_cbg_from_profile not in cbg_data:
            initial_cbg_from_profile = random.choice(list(cbg_data.keys()))
        agent = Agent(i, profile, initial_cbg_from_profile, cbg_data, cbg_summary, cbg_to_city, llm_client, traj_dir)
        agents.append(agent)
    logging.info(f"Initialized {len(agents)} agents.")
    # 主循环
    num_steps = config['num_steps']
    with tqdm(total=num_steps, desc="Simulation Steps", position=0, leave=True) as pbar_steps:
        for step in range(1, num_steps + 1):
            pbar_agents = tqdm(total=len(agents), desc=f"Step {step} Agents", position=1, leave=False)
            for agent in agents:
                agent.decide_next_location(step)
                pbar_agents.update(1)  # 更新 Agent 进度条
            pbar_agents.close()
            pbar_steps.update(1)  # 更新 Steps 进度条
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Simulation {run_dir} finished. Duration: {duration}")
    print(f"Simulation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()