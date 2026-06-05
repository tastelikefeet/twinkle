"""使用 Qwen3.5-4B-Condenser LoRA 对三类原始数据进行压缩的示例。

三个场景：
  1. Python 代码（短）
  2. 长度约 5120 字符的中文新闻文本
  3. 含混杂字符的网页 HTML 代码

除代码外的所有自然语言均为中文。压缩 LoRA 默认指向 ModelScope 上的
``ms://twinkle-kit/Qwen3.5-4B-Condenser``，即与 ``cookbook/exp/grpo_condensed.py``
所用 condenser 一致；可通过环境变量 ``LORA_PATH`` 覆盖。

启动方式::

    SAMPLER_GPUS=1 python cookbook/sample/sample.py
    SAMPLER_GPUS=2 python cookbook/sample/sample.py    # 张量并行
"""

import os
from typing import Any, Dict, List

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler

logger = get_logger()

# MODEL_ID = os.environ.get('MODEL_ID', 'output/condenser_ddp/step_44000')
MODEL_ID = 'Qwen/Qwen3.5-4B'
LORA_PATH = os.environ.get('LORA_PATH', 'ms://twinkle-kit/Qwen3.5-4B-Condenser')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))


# ──────────────────────────────────────────────────────────────────────
# Condenser 提示词（与训练时严格对齐，保留英文原文以匹配 LoRA 训练分布）
# ──────────────────────────────────────────────────────────────────────
CONDENSER_SYSTEM = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

Downstream model workflow:
Read your compressed output -> Decide whether needed info is in this block -> If yes -> Fetch original.

Therefore your compression MUST NOT lose major information from the source.

Output format:

```text
## Summary
Overview plus facts STRONGLY RELATED to the Query, stated explicitly.

## More
A collapsed index; expansion required to see specific information.
```

Rules:
1. Telegraphic style — drop function words ("the", "a", "is", "are", "of", ...); colons and commas mean "is" / "has".
2. Summary MUST contain the passage's primary topic + 2–4 concrete core facts drawn from the source (entities, numbers, dates, relations). If a Query is given, order Query-relevant facts first, but STILL include other core facts within the budget. A Query is an ORDERING HINT, NOT a filter.
3. Summary MUST NOT be meta-commentary about the Query. Forbidden patterns: "no X mention", "Query info: absent", "passage covers Y only", "does not contain ...", "no relevant info", or summaries that are only abstract category words like "structure/order/usage" with no facts. If the passage is unrelated to the Query, you still summarize the passage normally.
4. More is an INDEX of category keywords, NOT inline data. Enumerate what CAN be recovered from the source (e.g. "birthplace, death place, age"); do NOT paste dates/numbers/names inline. Make sure all category of useful facts are introduced here.
5. Output language MUST match the source language.
6. Do NOT fabricate. Do NOT omit major information. Any fact not in the source MUST NOT appear in your output.

Now begin.
"""

CONDENSER_USER = (
    'Downstream model will read your compressed block to decide whether to '
    'expand it. Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary '
    'about the Query (never write "Query info: absent", "no X mention", etc.); '
    'if the passage does not address the Query, still summarize the passage.\n\n'
    '## Query (ordering hint only — still summarize the whole passage)\n{query}\n\n'
    '## Target length\n'
    'Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars '
    '(~50% of the source). If core facts fit in far fewer chars, output fewer. '
    'Never exceed the ceiling.\n\n'
    '## Passage\n{text}')


# ──────────────────────────────────────────────────────────────────────
# 场景 1：Python 代码片段（Dijkstra 单源最短路）
# ──────────────────────────────────────────────────────────────────────
PY_QUERY = '这段代码的功能、方法名、出入参是什么？其他人如何调用？'
PY_PASSAGE = '''import heapq
from typing import Dict, List, Tuple


def dijkstra(graph: Dict[str, List[Tuple[str, float]]], src: str) -> Dict[str, float]:
    """Single-source shortest path on a non-negative weighted graph.

    Args:
        graph: adjacency list, ``graph[u] = [(v, w), ...]`` with ``w >= 0``.
        src:   source node id; must be a key of ``graph``.

    Returns:
        Mapping from node id to its shortest distance from ``src``;
        unreachable nodes get ``math.inf``.

    Time:  O((V + E) log V) via a binary heap.
    Space: O(V) for the distance map and the priority queue.
    """
    dist: Dict[str, float] = {node: float('inf') for node in graph}
    dist[src] = 0.0
    heap: List[Tuple[float, str]] = [(0.0, src)]
    visited: set = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if w < 0:
                raise ValueError(f'negative weight on edge {u}->{v}: {w}')
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (alt, v))
    return dist


if __name__ == '__main__':
    g = {
        'A': [('B', 1.0), ('C', 4.0)],
        'B': [('C', 2.0), ('D', 5.0)],
        'C': [('D', 1.0)],
        'D': [],
    }
    print(dijkstra(g, 'A'))
'''


# ──────────────────────────────────────────────────────────────────────
# 场景 2：长篇中文新闻（约 5120 字符）
# ──────────────────────────────────────────────────────────────────────
NEWS_QUERY = '本次峰会可以学习到什么，总结出什么经验？'
NEWS_PASSAGE = """2026年5月10日上午，为期三天的“全球低空经济创新峰会暨城市级低空示范走廊启用仪式”在合肥滨湖国际会展中心闭幕。会议由国家发展改革委、工业和信息化部、中国民用航空局共同主办，安徽省人民政府承办，吸引了来自三十六个国家和地区的一千二百余名代表，其中包括十七位省部级官员、四十二家飞行器整机企业代表、九十一家产业链上下游企业、二十三家科研院所及七家国际行业协会。会议公布了《低空经济创新发展指数（2026）》、《城市低空运行规则白皮书（试行版）》和《低空安全能力评估通用框架》等三份核心文件，明确将合肥、深圳、苏州、广州、成都、青岛六城列为首批“低空经济综合改革试验区”，并宣布合肥滨湖至庐阳总长四十六公里的“环城低空示范走廊”当日正式投入运行。

按照规划，环城低空示范走廊由九条主干航线和十六条支线组成，主干航线最低离地高度一百二十米、最高三百米，支线最低六十米、最高一百八十米；全网部署一百八十二个固定起降点和六十座可移动塔台，覆盖政务、医疗、应急、物流、低空旅游、城市巡检等十类典型业务场景。走廊采用“一张网、两套链、三层防”的运行架构，统一接入安徽省低空运行管理平台，平台部署三百台分布式边缘节点和两套异地灾备数据中心，单日峰值并发架次设计能力为六千架，支持十秒级动态空域调整与三十毫秒级冲突告警。运行首日上午即完成首班医疗血液配送、首班跨区低空通勤、首班高速公路应急救援与首班低空观光飞行等四项标志性任务。

国家发展改革委副主任周楠在主旨演讲中表示，低空经济作为我国正在加快培育的战略性新兴产业，2025年市场规模已突破六千八百亿元，年复合增长率连续三年保持在百分之三十二以上；按照《低空经济创新发展指数（2026）》预测，到2030年市场规模有望突破三万亿元，将带动直接就业岗位约二百二十万个、间接就业八百万个。她强调，下一阶段的政策重点将集中在三件事上：一是推动空域分类改革落地，将三百米以下空域使用审批权限下放至省级；二是建立全国统一的低空飞行身份认证体系，以“一码通飞”形式整合飞行器编号、运营资质、保险信息；三是加快建设“低空气象－通信－导航－监视”四张网，2027年前在三十座中心城市完成基础设施全覆盖。

中国民用航空局副局长邵岩晖介绍，新版《城市低空运行规则白皮书（试行版）》对运行主体提出了五项硬性要求：飞行器须取得型号合格证或试飞许可、运行人须建立安全管理体系并通过年度审核、机长须持有相应等级有效执照、第三方责任险保额单架不得低于人民币五百万元、城市核心区运行须接入城市低空数据共享平台。白皮书首次明确了无人机与有人机融合运行规则，规定融合空域内电子围栏精度优于五米、上传频率不低于每秒十次、应急避让响应时间不大于八百毫秒。试行版将在合肥、深圳、苏州先行实施六个月，2027年1月起在六个综合改革试验区全面推广。

本次峰会期间，共有四十八家整机与零部件企业进行了集中签约，签约总金额三百一十七亿元人民币。其中，亿航智能与合肥市政府就eVTOL航空枢纽建设达成战略合作，未来三年将在合肥落地两座垂直起降中心、一座飞行器维修工厂；峰飞航空V2000CG无人货运飞机宣布与京东物流、顺丰速运组建“低空干线物流联盟”，2026年内开通合肥-武汉、合肥-南京两条三百公里级日常货运航线；时的科技E20 eVTOL正式获得中国民用航空局型号合格审定（TC）受理通知，成为国内第二个进入TC审定阶段的国产载人eVTOL机型；中国电信、中国移动联合发布“低空通信定制网络”，提供基于5G-A的厘米级定位与十毫秒级时延切片服务，首批接入合肥、深圳、苏州三市示范走廊。

中国科学技术大学、北京航空航天大学、南京航空航天大学、中国航发湖南动力机械研究所等四家单位联合发布了五项关键技术成果。其中，中科大研制的“星臂II号”分布式电推进系统单机连续可靠工作时长突破六千小时，能量密度达到每千克四百二十瓦时；北航团队公布国内首套适用于城市楼宇间复杂气流环境的“激流-3”自主感知与避障算法，已在合肥CBD连续完成八千架次实飞验证，避障成功率达百分之九十九点九七；南航发布的“穹顶”机载多源融合定位单元在GPS拒止环境下可实现亚米级定位，精度优于现有民用产品三倍。

国际合作方面，中欧双方在峰会上签署《低空运行互认合作备忘录》，约定2027年起对各自认证的两吨级以下载人电动飞行器互相承认型号合格证基础部分，争议技术指标通过联合评审解决。中国与阿联酋、新加坡、巴西、德国、日本五国民航主管部门签署双边谅解备忘录，覆盖低空气象数据互通、跨境物流走廊试点、飞行员资质互认三个方向。世界经济论坛代表在致辞中评价，合肥示范走廊是“迄今为止全球规模最大、运营规则最完整的城市级低空融合试验场”。

为保障示范走廊安全运行，安徽省专门组建了“低空安全联合运行中心”，由民航华东空管局、安徽省公安厅、应急管理厅、气象局以及合肥市政府五方常态派员，实行7×24小时值守。运行中心配备六十四套全向相控阵雷达、九十二套低空ADS-B接收机和一百二十组光电跟踪设备，可对覆盖空域内大于0.05平方米的目标进行毫秒级追踪；同时部署了五十架次自动巡查无人机和两架有人机巡查直升机，对低慢小目标实行混合编队拦截。运行首日，中心累计处置告警事件二十三起，其中误闯入九起、设备失联六起、超高飞行四起、外部气象突变三起、未授权改航一起，全部在三分钟内完成处置。

针对普通市民关心的应用场景，主办方在滨湖国际会展中心外侧搭建了占地约一万二千平方米的“低空生活体验区”。市民可通过现场或“合肥低空”小程序预约低空观光（合肥环城线，单程二十分钟，票价人民币二百九十八元）、低空通勤（滨湖至合肥南站，单程八分钟，票价九十八元）、低空配送（三公里内三十分钟达，订单费十二元）和低空应急医疗演示等四项体验。仪式当日预约平台一上线即满负荷运转，截至当天下午五点，累计提交订单超过一万一千笔，其中观光类占百分之六十四、通勤类占百分之二十二、配送类占百分之十三。

投融资方面，峰会同期举行的“低空产业投资人之夜”披露：2025年我国低空领域股权融资总额突破七百八十亿元，同比增长百分之七十二，融资轮次主要集中在A轮至C轮，平均单笔金额一点二亿元；其中飞行器整机、电池电机电控、空管软件三类标的占比分别为百分之四十一、百分之二十三、百分之十八。安徽省产业投资集团联合中国国新基金、中信产业基金、深创投、合肥兴泰金融等十家机构发起设立总规模二百亿元的“低空经济母基金”，首期规模六十亿元，重点投向飞行器适航取证、低空通信导航、城市运营平台与高端材料四个方向，单项目投资上限三亿元，预计三年内完成对外投放。

人才培养方面，国家民航局、教育部、人力资源和社会保障部联合发布《低空飞行人员培养行动计划（2026-2030）》，明确到2030年累计培养低空领域专业人才四十万人，其中eVTOL机长六万人、地面运行控制员八万人、无人机系统工程师十二万人、低空气象与运行支持人员六万人、产业链高端研发人员八万人。中国民航大学、中国民用航空飞行学院、合肥工业大学、深圳职业技术大学等十六所院校将于2026年秋季学期同步开设“低空运行与管理”本科专业和“无人飞行器系统工程”研究生方向，前两年招生总规模约六千八百人，并实行校企双导师制。

法治保障方面，《中华人民共和国低空空域使用管理条例（草案）》已于4月底完成第二轮社会公开征求意见，预计2026年下半年提交全国人大常委会审议。条例（草案）首次以法律形式确立“分类分级、责任清晰、动态管理”的空域使用原则，明确300米以下非管制空域备案准入、300米至1000米管制空域许可准入；规定运行人对所造成的人身、财产损害承担无过错责任，强制责任险最低保额按机型分为单架人民币三百万、五百万、一千万三档；对违规飞行的行政处罚上限从原《民用航空法》的人民币十万元提高至人民币一百万元，构成犯罪的依法追究刑事责任。

民众体验环节中，本报记者亲自试乘了由亿航EH216-S执飞的合肥环城观光航线。从滨湖国际会展中心垂直起降平台起飞，飞行器在二十秒内攀升至一百八十米高度，随后沿环城西线向北巡航，途经合肥南站、政务区、合肥植物园等地标，全程巡航速度九十公里每小时，最高速度一百一十公里每小时，舱内噪音实测六十六分贝、相当于普通会议室水平；地面起降阶段振动幅度小于零点二G，乘坐感受平稳。值得一提的是，飞行器全程由地面无人值守，机舱内仅有四枚乘客座位与一台显示飞行参数和航线进度的十英寸触控屏，乘客可一键切换中文、英文、日文三种语音解说。

技术展望部分，多位专家在分论坛中表达共识：未来五年制约低空经济规模化的关键不是飞行器性能，而是“运行密度天花板”——即在城市核心区如何把单位空域、单位时间内的安全飞行架次密度从当前的十架次每平方公里每小时提升到二百架次。中国工程院院士王建宇指出，要突破这一瓶颈必须解决三个核心问题：一是低慢小目标的全天候、全气象、全城域感知；二是冲突探测与解脱算法在高密度场景下的实时性，目标响应时间需压缩到二十毫秒以内；三是空地一体化通信网络的可用性，必须达到5个9（99.999%）的可靠度。预计这些核心技术将在“十四五”末取得阶段性突破，并在“十五五”实现产业化推广。

区域协同与产业布局方面，本次峰会同期发布了《中国低空经济产业布局白皮书（2026）》，首次以全国六十三个重点城市为样本，对产业链上下游进行了详尽画像。白皮书揭示，现阶段我国低空产业已初步形成“三极多点”的空间格局：长三角以合肥、南京、苏州、上海、杭州五市为核心，重点发展eVTOL整机、高端重载无人机与运营平台，产业营收占全国百分之三十五；珠三角以深圳、广州为核心，重点发展消费级与商业级无人机、低空物流，产业营收占百分之二十九；成渝地区以成都、重庆为核心，重点发展航空发动机、错复材料与航电系统，占百分之十三；其余千亿级“多点”包括青岛、沈阳、西安、武汉、长沙五个区域中心。白皮书同时提示，中西部低空产业发展仍存在不平衡问题，需重点加强边境州、边境县及山区、草原地区的低空基础设施补齐，预计这些区域有3200多个乡镇需新增低空起降点。为此，发改委将联合农业农村部、国务院应急管理部启动专项补助，中央财政三年内安排专项资金一百五十亿元。

闭幕仪式上，安徽省政府宣布将在2026年内追加投资八十亿元用于二期走廊建设，二期工程将向北延伸至淮南、向西延伸至六安，总长度从四十六公里扩展到一百九十六公里，预计2027年底前贯通。下一步合肥还将牵头编制《长三角低空一体化运行总体方案》，推动沪苏浙皖四省市在2028年前实现“一码通飞、一卡结算、一平台调度”的跨省域低空运行体系。中国民用航空局表示，将在2026年第三季度发布《国家低空经济发展中长期规划（2026-2035）》，明确未来十年的总体目标、重点任务、保障措施与考核机制。本次峰会的全部技术文件、签约项目清单和示范走廊运行实时数据将在“中国低空经济服务网”同步公开。据主办方最后公布的统计，为期三天的峰会共举办主论坛一场、高端对话三场、专题分论坛二十三场、企业发布会十六场，现场展示飞行器与装备总计一百八十八架（套），其中eVTOL三十二架、重载无人货运机二十七架、高端作业型无人机六十五架、空管与低空助航装备五十四套；累计进场观众超过二十三万人次，现场达成预订订单十二万五千余笔，为合肥市仪式期间酒店入住率、餐饮营业额分别带来同比百分之四十二与百分之三十六的增长。与会代表普遍评价，本届峰会首次实现了“会议、试点、产业、民生”四者同场并进，将原本存在于不同部门、不同会议、不同后续跨年的不同工作压缩为一个集中阶段，明显提高了这轮低空经济发展的政策准备度与社会可见度。据估计，在后续三个月内，首批六个“低空经济综合改革试验区”均将起步运行走廊与试点业务，预计2027年上半年可被看到首轮可复制、可推广的运行经验与产业样本。
"""


# ──────────────────────────────────────────────────────────────────────
# 场景 3：含混杂字符的网页 HTML（电商商品详情页）
# ──────────────────────────────────────────────────────────────────────
HTML_QUERY = '这段html代码的结构如何？如何使用js如何对接？'
HTML_PASSAGE = """<!DOCTYPE html>
<html lang="zh-CN" data-spm="product-detail">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>云端Air Pro 13 笔记本电脑（2026款）| TechMart 数码旗舰店</title>
  <meta name="description" content="云端Air Pro 13，搭载自研M3-Pro芯片，14核CPU+18核GPU，售价¥9,999，限时8折，赠AppleCare+">
  <meta property="og:price:amount" content="9999.00">
  <meta property="og:price:currency" content="CNY">
  <link rel="canonical" href="https://shop.techmart.cn/p/yda-pro-13-2026">
  <script>window.__INITIAL_STATE__ = {"sku":"YDA-PRO-13-2026","stock":237,"region":"CN-AH"};</script>
  <style>
    .price del{color:#999;text-decoration:line-through;margin-right:8px;}
    .price strong{color:#e60012;font-size:32px;font-weight:700;}
    .badge{background:#ff4d4f;color:#fff;padding:2px 6px;border-radius:4px;font-size:12px;}
  </style>
</head>
<body>
  <header class="nav"><a href="/">首页</a> &gt; <a href="/c/laptop">笔记本电脑</a> &gt; 云端Air Pro 13</header>
  <main>
    <h1>云端Air Pro 13&nbsp;<span class="badge">2026款 · 全网首发</span></h1>
    <p class="brand">品牌：云端 (Yunduan)&nbsp;|&nbsp;型号：YDA-PRO-13-2026&nbsp;|&nbsp;颜色：星空银 / 深空灰 / 沙漠金</p>
    <div class="price">
      <del>¥12,499.00</del>
      <strong>¥9,999.00</strong>
      <span>立省 ¥2,500，限时48小时</span>
    </div>
    <ul class="spec">
      <li>处理器：自研 M3-Pro，14核CPU @ 3.6GHz / 18核GPU / 16核NPU（35 TOPS）</li>
      <li>内存：18GB LPDDR5X-7500（统一内存架构）</li>
      <li>存储：512GB / 1TB / 2TB NVMe SSD（最高读取 7,400 MB/s）</li>
      <li>屏幕：13.6&Prime; Liquid Retina，2,560×1,664，600 nits 峰值，DCI-P3 100%</li>
      <li>电池：72Wh，续航最长 18h（本地视频播放）</li>
      <li>重量：1.24 kg&nbsp;|&nbsp;厚度：11.3 mm</li>
      <li>接口：2× Thunderbolt 5、1× HDMI 2.1、1× MagSafe 3、1× 3.5mm 耳机</li>
      <li>无线：Wi-Fi 7（802.11be）、蓝牙 5.4、UWB</li>
    </ul>
    <p>赠品（前 100 名下单）：原装 65W GaN 电源、Type-C&rarr;HDMI 2.1 转换线、防泼溅键盘膜、AppleCare+ 1 年延保。</p>
    <div class="promo">!! 限时优惠：叠加云端校园券 ¥500 + 以旧换新最高补贴 ¥1,200 !!</div>
    <section class="qa">
      <h2>常见问题</h2>
      <p>Q: 是否支持 Windows 11 ARM 双系统？&nbsp;A: 不支持，但可通过 Parallels Desktop 19 虚拟运行。</p>
      <p>Q: 发货时效？&nbsp;A: 现货 24 小时内发出，安徽/江苏/浙江/上海次日达。</p>
    </section>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Product","name":"云端Air Pro 13",
     "sku":"YDA-PRO-13-2026","brand":{"@type":"Brand","name":"Yunduan"},
     "offers":{"@type":"Offer","price":9999.00,"priceCurrency":"CNY",
               "availability":"https://schema.org/InStock"}}
    </script>
  </main>
  <footer>&copy; 2026 TechMart Inc. 沪ICP备2021xxxx号 &middot; 京公网安备 31010102xxxxxx号</footer>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# 场景 4：复杂异常处理 Python 代码（支付下单处理器）
# ──────────────────────────────────────────────────────────────────────
# 故意混入多种异常处理风格：自定义异常树、链式抛出、bare except 反模式、
# 资源未关闭、重试与回退、上下文管理器、suppress、finally 重写返回值等。
EXCEPTIONS_QUERY = (
    '这段支付下单代码的异常处理设计了哪些模式、踩了哪些反模式坑、'
    '可以总结出哪些最佳实践和教训？')
EXCEPTIONS_PASSAGE = '''import json
import logging
import time
from contextlib import suppress
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---- Domain exception hierarchy ----
class PaymentError(Exception):
    """Base for all payment-domain errors."""


class TransientPaymentError(PaymentError):
    """Retryable: timeout, 5xx, network flap."""


class PermanentPaymentError(PaymentError):
    """Non-retryable: 4xx, invalid card, fraud block."""


class IdempotencyConflict(PermanentPaymentError):
    """Idempotency-Key reused with a different request body."""


class OrderRepository:
    def __init__(self, conn):
        self.conn = conn  # NOTE: caller owns the connection lifetime.

    def begin(self):
        self.conn.execute('BEGIN')

    def commit(self):
        self.conn.execute('COMMIT')

    def rollback(self):
        # Anti-pattern guard: swallow rollback errors to not mask the original.
        with suppress(Exception):
            self.conn.execute('ROLLBACK')

    def mark_paid(self, order_id: str, txn_id: str):
        self.conn.execute(
            'UPDATE orders SET status=?, txn_id=? WHERE id=?',
            ('PAID', txn_id, order_id))


def _call_gateway(url: str, body: dict, idem_key: str, timeout: float = 3.0):
    """Single HTTP call. Translates transport errors into the domain hierarchy."""
    try:
        resp = requests.post(
            url, json=body, timeout=timeout,
            headers={'Idempotency-Key': idem_key})
    except requests.Timeout as e:
        raise TransientPaymentError(f'gateway timeout: {url}') from e
    except requests.ConnectionError as e:
        raise TransientPaymentError(f'gateway unreachable: {url}') from e

    if 500 <= resp.status_code < 600:
        raise TransientPaymentError(f'gateway 5xx: {resp.status_code}')
    if resp.status_code == 409:
        # Same key, different body — caller bug, never retry.
        raise IdempotencyConflict(f'idem-key reused: {idem_key}')
    if 400 <= resp.status_code < 500:
        raise PermanentPaymentError(
            f'gateway 4xx: {resp.status_code} body={resp.text[:200]}')

    try:
        return resp.json()
    except json.JSONDecodeError as e:
        # Server claimed 2xx but body is junk; treat as transient — gateway bug.
        raise TransientPaymentError('gateway returned non-JSON 2xx') from e


def charge_with_retry(url: str, body: dict, idem_key: str,
                      max_attempts: int = 4) -> dict:
    """Exponential backoff. ONLY retries TransientPaymentError."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_gateway(url, body, idem_key)
        except TransientPaymentError as e:
            last_exc = e
            sleep_s = min(2 ** (attempt - 1), 8)
            logger.warning('charge attempt %d/%d failed: %s; sleep %ss',
                           attempt, max_attempts, e, sleep_s)
            time.sleep(sleep_s)
        # PermanentPaymentError intentionally propagates immediately.
    assert last_exc is not None  # for type-checker; loop guarantees this.
    raise TransientPaymentError(
        f'exhausted {max_attempts} attempts') from last_exc


def place_order(repo: OrderRepository, order_id: str, body: dict,
                gateway_url: str) -> bool:
    """End-to-end order placement. Returns True on success.

    Lessons embedded in this body:
    - Idempotency-Key is derived from order_id (NOT a random uuid per attempt)
      so retries hit the gateway as the same logical request.
    - Catch broad exceptions ONLY at the outermost trust boundary, never
      inside the loop.
    - The bare-except below (legacy debugger pattern) IS A BUG — it suppresses
      KeyboardInterrupt and SystemExit; left here intentionally to be flagged.
    """
    idem_key = f'order:{order_id}'
    repo.begin()
    try:
        receipt = charge_with_retry(gateway_url, body, idem_key)
        repo.mark_paid(order_id, receipt['txn_id'])
        repo.commit()
        return True
    except IdempotencyConflict:
        # Loud: indicates a programming error upstream.
        repo.rollback()
        logger.exception('idempotency conflict on %s', order_id)
        raise
    except PermanentPaymentError as e:
        # Expected business failure: rollback and surface a typed error.
        repo.rollback()
        logger.warning('order %s rejected by gateway: %s', order_id, e)
        return False
    except TransientPaymentError as e:
        # Retries already exhausted; do not swallow.
        repo.rollback()
        logger.error('order %s transient failure: %s', order_id, e)
        raise
    except:  # noqa: E722  -- ANTI-PATTERN, intentionally left for review.
        # Catches KeyboardInterrupt / SystemExit / MemoryError too. Bad.
        repo.rollback()
        logger.exception('unexpected failure on %s', order_id)
        return False
    finally:
        # Anti-pattern: returning from finally would swallow exceptions; we DO NOT
        # return here. Only release locks / log timing.
        logger.debug('place_order(%s) finished', order_id)
'''


# ──────────────────────────────────────────────────────────────────────
# 场景 5：混合服务日志（正常 / 不规则 / 异常 三类掺杂）
# ──────────────────────────────────────────────────────────────────────
# 目标：考察压缩模型能否在大量噪声中突出真正的故障信号。
# Summary 应聚焦异常（ERROR/FATAL/堆栈），常规心跳/健康检查应被压成索引词。
LOGS_QUERY = (
    '这堆服务日志里发生了哪些独立故障？要求把每一条 ERROR/FATAL/异常都作为独立条目列出来，'
    '附上其专属标识（订单号 ORD-xxx / 退款键 refund:xxx / Pod 名 / 主机名 / 证书 CN / '
    'PagerDuty 单号 / Kafka topic+partition / trace_id 等），同名不同实例必须分别列出，不得合并；'
    '正常心跳和健康检查只需在末尾用一两句索引带过，不要展开。')
LOGS_PASSAGE = '''2026-05-28T03:14:00.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=412MB cpu=3.1%
2026-05-28T03:14:00.118Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:00.402Z INFO  [order-svc-12] POST /orders 200 27ms user=u_88231 amount=199.00
2026-05-28T03:14:00.512Z INFO  [api-gw-7f9c]  GET /v1/products?cat=3 200 9ms
2026-05-28T03:14:00.690Z DEBUG [cache-3a1]    redis.get key=sess:9ab miss=false ttl=512s
2026-05-28T03:14:00.731Z INFO  [order-svc-09] POST /orders 200 18ms user=u_88232 amount=12.49
May 28 03:14:00 host-edge-03 kernel: [13929847.221] TCP: request_sock_TCP: Possible SYN flooding on port 443. Sending cookies.
2026-05-28T03:14:00.812Z INFO  [search-svc]   query took=7ms hits=104 q="鼠标"
{"ts":"2026-05-28T03:14:00.901Z","lvl":"info","svc":"recom","msg":"model v17 served","qps":3120,"p99_ms":42}
2026-05-28T03:14:01.005Z DEBUG [cache-3a1]    redis.get key=sess:abc miss=false ttl=287s
2026-05-28T03:14:01.122Z INFO  [api-gw-7f9c]  GET /v1/products?cat=12 200 14ms
172.18.4.21 - - [28/May/2026:03:14:01 +0000] "GET /static/app.js HTTP/1.1" 200 84217 "-" "Mozilla/5.0" rt=0.003
172.18.4.22 - - [28/May/2026:03:14:01 +0000] "GET /static/app.css HTTP/1.1" 304 0 "-" "Mozilla/5.0" rt=0.001
2026-05-28T03:14:01.480Z DEBUG [order-svc-12] cart.compute total=199.00 items=3 user=u_88231 promo=SPRING10
2026-05-28T03:14:01.611Z INFO  [audit]        write event=login user=u_88245 ip=203.0.113.44 ua=ios/9.2.1
2026-05-28T03:14:01.799Z WARN  [order-svc-12] payment.gateway latency=812ms threshold=500ms attempt=1/3
2026-05-28T03:14:01.901Z WARN  [payment-svc]  retry-budget remain=87/100 window=60s
03:14:02 (??) [????]  partial frame: \x7f\x45\x4c\x46\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00 ... <2174 bytes dropped, recovery=skip>
[bgworker] !! 一中一英插错位 !! sync orders snapshot 开始， shard=7  records=124882
2026-05-28T03:14:02.044Z INFO  [order-svc-12] POST /orders 200 31ms user=u_88245 amount=49.90
2026-05-28T03:14:02.211Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:02.310Z WARN  [payment-svc]  gateway timeout retry_after=2s url=https://pay.acme.io/charge idem=order:ORD-44871
2026-05-28T03:14:02.402Z DEBUG [search-svc]   parsed query bm25_terms=[laptop,gaming] filters={"price":[null,2000]}
2026-05-28T03:14:02.510Z INFO  [recom]         shard-warmup done shard=5 took=88ms
2026-05-28T03:14:02.612Z ERROR [payment-svc]  charge failed: TransientPaymentError: gateway timeout: https://pay.acme.io/charge
  Traceback (most recent call last):
    File "payments/client.py", line 88, in _call_gateway
      resp = requests.post(url, json=body, timeout=3.0, ...)
    File "requests/api.py", line 115, in post
      raise Timeout("HTTPSConnectionPool: Read timed out.")
  requests.exceptions.Timeout: HTTPSConnectionPool(host=\'pay.acme.io\', port=443): Read timed out. (read timeout=3.0)
2026-05-28T03:14:02.613Z INFO  [payment-svc]  retry attempt=2/3 backoff=2s key=order:ORD-44871
2026-05-28T03:14:02.701Z INFO  [api-gw-7f9c]  GET /v1/products/9001 200 11ms
2026-05-28T03:14:02.812Z INFO  [order-svc-09] POST /orders 200 22ms user=u_88251 amount=320.00
2026-05-28T03:14:02.901Z DEBUG [cache-3a1]    redis.scan cursor=0 count=200 match=sess:*
2026-05-28T03:14:03.000Z INFO  [api-gw-7f9c]  heartbeat ok rss=414MB cpu=3.4%
2026-05-28T03:14:03.044Z INFO  [order-svc-12] GET /orders/ORD-44870 200 4ms
???GARBLED??? ½üÏí hé shì ?? mq.consumer offset=49281234 lag=??? 中间被截断
May 28 03:14:03 host-db-master postgres[2174]: LOG:  checkpoint starting: time
2026-05-28T03:14:03.244Z INFO  [recom]         a/b experiment exp_id=ex-991 traffic=10% bucket=v17
[2026/05/28 03:14:03.488] log格式错位 -- 商品库存同步开始 batch=512
2026-05-28T03:14:03.601Z INFO  [order-svc-12] POST /orders 200 19ms user=u_88260 amount=4.99
2026-05-28T03:14:03.812Z INFO  [api-gw-7f9c]  GET /v1/products?cat=8 200 7ms
2026-05-28T03:14:03.901Z DEBUG [cache-3a1]    redis.get key=sess:def miss=false ttl=600s
2026-05-28T03:14:04.117Z INFO  [inventory]    sync done batch=512 ok=512 fail=0 took=629ms
2026-05-28T03:14:04.230Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:04.401Z INFO  [order-svc-12] POST /orders 200 24ms user=u_88263 amount=88.00
2026-05-28T03:14:04.602Z ERROR [payment-svc]  charge failed: PermanentPaymentError: gateway 4xx: 402 body={"code":"INSUFFICIENT_FUNDS","order":"ORD-44871","user":"u_88251","reason":"card balance below required amount","trace_id":"pgw-c1a2b3d4"}
2026-05-28T03:14:04.604Z WARN  [order-svc-12] order ORD-44871 rejected, status=PAYMENT_FAILED user=u_88251
2026-05-28T03:14:04.707Z INFO  [audit]        write event=order.rejected order=ORD-44871 reason=insufficient_funds
2026-05-28T03:14:04.811Z INFO  [recom]         model v17 served qps=3140 p99_ms=44
2026-05-28T03:14:04.901Z INFO  [api-gw-7f9c]  POST /v1/cart 200 12ms user=u_88277
2026-05-28T03:14:05.001Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:05.103Z DEBUG [search-svc]   parsed query bm25_terms=[laptop] filters={}
2026-05-28T03:14:05.220Z INFO  [order-svc-09] POST /orders 200 16ms user=u_88281 amount=33.30
2026-05-28T03:14:05.330Z INFO  [search-svc]   query took=12ms hits=37 q="laptop"
2026-05-28T03:14:05.501Z INFO  [api-gw-7f9c]  GET /v1/products?cat=5 200 9ms
2026-05-28T03:14:05.612Z DEBUG [cache-3a1]    redis.del key=sess:xyz removed=1
2026-05-28T03:14:05.778Z DEBUG [cache-3a1]    redis.set key=sess:def ttl=600s size=384B
>>>>> RAW FRAME @ tcp://10.0.6.4:9092 - kafka consumer rebalance event - members=[c1,c2,c3,c4] generation=812
2026-05-28T03:14:05.901Z INFO  [audit]        write event=cart.update user=u_88290 items=4
2026-05-28T03:14:06.012Z INFO  [order-svc-12] POST /orders 200 22ms user=u_88260 amount=12.00
=== unstructured dump @ 03:14:06.4 ===  conn_pool active=48/64 idle=11 waiting=5 (warn>40) db=orders_master
2026-05-28T03:14:06.412Z WARN  [db-pool]      pool nearly exhausted: active=48 max=64 waiting=5 db=orders_master
2026-05-28T03:14:06.500Z WARN  [db-pool]      slow query detected took=812ms sql="UPDATE orders SET status=$1 WHERE shard=$2 AND ts<$3" rows=4188
2026-05-28T03:14:06.612Z INFO  [recom]         shard-warmup done shard=6 took=92ms
2026-05-28T03:14:06.711Z INFO  [search-svc]   query took=8ms hits=21 q="keyboard"
2026-05-28T03:14:06.901Z DEBUG [cache-3a1]    redis.get key=sess:0a1 miss=true ttl=0s
2026-05-28T03:14:07.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=418MB cpu=4.0%
2026-05-28T03:14:07.118Z INFO  [order-svc-12] POST /orders 200 28ms user=u_88277 amount=88.50
2026-05-28T03:14:07.232Z INFO  [api-gw-7f9c]  GET /v1/orders/ORD-44871 200 5ms
2026-05-28T03:14:07.402Z WARN  [order-svc-09] retry payment user=u_88251 attempt=2 idem=order:ORD-44871
2026-05-28T03:14:07.612Z ERROR [order-svc-09] retry blocked: idempotency-conflict, charge already permanently failed
2026-05-28T03:14:07.910Z ERROR [order-svc-12] DB write failed: psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "orders_pkey"
  DETAIL:  Key (id)=(ORD-44875) already exists.
  CONTEXT: COPY orders, line 1
  Traceback (most recent call last):
    File "order/repo.py", line 142, in insert
      cur.execute(SQL_INSERT, payload)
    File "psycopg2/cursor.py", line 234, in execute
      self._execute_impl(query, vars)
  psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "orders_pkey"
  -- query: INSERT INTO orders(id, user_id, amount, status, shard, ts) VALUES ($1,$2,$3,$4,$5,now())
  -- params: ('ORD-44875', 'u_88278', 88.50, 'NEW', 7)
2026-05-28T03:14:07.912Z ERROR [order-svc-12] POST /orders 500 187ms user=u_88278 trace_id=4f1c9a2b
2026-05-28T03:14:07.998Z INFO  [audit]        write event=order.failed order=ORD-44875 reason=duplicate_key
???? 5月28日 03:14:08 中间件告警: db-pool active=51 (中文告警) ?? 原始编码 GBK?
2026-05-28T03:14:08.044Z INFO  [search-svc]   query took=14ms hits=66 q="mouse"
2026-05-28T03:14:08.117Z INFO  [api-gw-7f9c]  POST /v1/login 200 33ms user=u_88290
2026-05-28T03:14:08.232Z DEBUG [cache-3a1]    redis.expire key=sess:5cd ttl=900s ok=1
2026-05-28T03:14:08.330Z INFO  [api-gw-7f9c]  GET /v1/products/12 200 6ms
2026-05-28T03:14:08.444Z WARN  [k8s-probe]    livenessProbe pod=worker-22 statusCode=200 took=18ms (slow_threshold=10ms)
2026-05-28T03:14:08.601Z INFO  [recom]         model v17 served qps=3155 p99_ms=46
2026-05-28T03:14:08.722Z DEBUG [order-svc-12] cart.compute total=88.50 items=2 user=u_88278 promo=-
2026-05-28T03:14:08.811Z INFO  [api-gw-7f9c]  GET /v1/products?cat=2 200 8ms
2026-05-28T03:14:08.991Z WARN  [jvm-gc]       G1 Old Gen pause=412ms heap_after=6.8G/8G   # crosses 80% headroom, full-gc=0
2026-05-28T03:14:09.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=421MB cpu=4.2%
2026-05-28T03:14:09.122Z WARN  [jvm-gc]       G1 Old Gen pause=508ms heap_after=7.2G/8G   # heap pressure increasing
????-??-??T??:??:??.???Z ??? [????] (timestamp parse failed) raw="flush queue depth=131072 lag=4.2s svc=worker-22"
2026-05-28T03:14:09.401Z WARN  [worker-22]    allocation slow: requested=512MB available=178MB triggering full GC
2026-05-28T03:14:09.601Z WARN  [worker-22]    full GC initiated: heap=7.8G/8G live=7.6G
2026-05-28T03:14:09.700Z FATAL [worker-22]    java.lang.OutOfMemoryError: Java heap space
  at com.acme.order.Aggregator.fold(Aggregator.java:88)
  at com.acme.order.Aggregator.fold(Aggregator.java:71)
  at com.acme.order.Aggregator.run(Aggregator.java:42)
  at com.acme.metric.WindowSink.flush(WindowSink.java:204)
  at com.acme.metric.WindowSink$Worker.runOnce(WindowSink.java:158)
  at com.acme.metric.WindowSink$Worker.run(WindowSink.java:121)
  at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1136)
  at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:635)
  at java.base/java.lang.Thread.run(Thread.java:840)
  Suppressed: java.lang.OutOfMemoryError: GC overhead limit exceeded
    at com.acme.metric.RollupBuffer.append(RollupBuffer.java:312)
    ... 8 more
2026-05-28T03:14:09.702Z FATAL [worker-22]    process exiting, dumping heap to /var/log/heap/worker-22-1748400849.hprof (size≈6.7GB)
May 28 03:14:09 host-edge-03 kernel: [13929856.881] Out of memory: Killed process 39087 (java) total-vm:9421248kB, anon-rss:7842316kB, file-rss:412kB, shmem-rss:0kB, UID:1000 pgtables:16448kB oom_score_adj:0
2026-05-28T03:14:10.001Z ERROR [supervisor]   child worker-22 exited code=137 (OOMKilled) restart_in=5s
2026-05-28T03:14:10.122Z WARN  [api-gw-7f9c]  upstream worker-22 marked DOWN, routing to worker-{19,20,21,23}
2026-05-28T03:14:10.220Z INFO  [k8s-probe]    pod worker-22 phase=Failed reason=OOMKilled lastTerm="node memory pressure"
2026-05-28T03:14:10.301Z WARN  [k8s-probe]    node node-edge-03 conditions: MemoryPressure=true (since 2026-05-28T03:14:08Z)
2026-05-28T03:14:10.422Z INFO  [audit]        write event=worker.crashed worker=worker-22 reason=oom
2026-05-28T03:14:10.611Z INFO  [recom]         shard-rebalance triggered cause=worker-22-down shards-moved=2
2026-05-28T03:14:10.812Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:11.000Z INFO  [api-gw-7f9c]  heartbeat ok rss=423MB cpu=4.5%
2026-05-28T03:14:11.118Z INFO  [order-svc-12] POST /orders 200 33ms user=u_88290 amount=320.00
2026-05-28T03:14:11.244Z DEBUG [cache-3a1]    redis.get key=sess:7ef miss=false ttl=120s
2026-05-28T03:14:11.401Z INFO  [order-svc-09] POST /orders 200 19ms user=u_88299 amount=7.20
2026-05-28T03:14:11.512Z INFO  [api-gw-7f9c]  POST /v1/cart 200 11ms user=u_88301
=========================== TRUNCATED LOG SECTION (~14KB removed: 217 routine entries, 0 ERROR/FATAL) ===========================
2026-05-28T03:14:11.660Z INFO  [search-svc]   query took=9ms hits=12 q="keyboard"
2026-05-28T03:14:11.802Z DEBUG [order-svc-12] cart.compute total=320.00 items=5 user=u_88290 promo=VIP20
2026-05-28T03:14:11.901Z INFO  [recom]         model v17 served qps=3120 p99_ms=43
2026-05-28T03:14:12.001Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:12.122Z INFO  [api-gw-7f9c]  GET /v1/products?cat=1 200 7ms
2026-05-28T03:14:12.244Z WARN  [auth-svc]     kafka producer in-flight=80/100 approaching limit
2026-05-28T03:14:12.402Z ERROR [auth-svc]     kafka publish failed topic=user.login partition=3 err=NotLeaderForPartitionException broker=broker-2:9092 retry=1/5
  java.util.concurrent.ExecutionException: org.apache.kafka.common.errors.NotLeaderForPartitionException: This server is not the leader for that topic-partition.
    at org.apache.kafka.clients.producer.internals.FutureRecordMetadata.valueOrError(FutureRecordMetadata.java:101)
    at org.apache.kafka.clients.producer.KafkaProducer$FutureFailure.<init>(KafkaProducer.java:1356)
2026-05-28T03:14:12.404Z WARN  [auth-svc]     fallback to broker-1:9092, in-flight=84 will be retried
2026-05-28T03:14:12.522Z INFO  [auth-svc]     metadata refresh ok partitions=24 leaders={0:b1, 1:b1, 2:b3, 3:b1, ...}
2026-05-28T03:14:12.611Z INFO  [auth-svc]     kafka publish ok topic=user.login partition=3 offset=49281999 took=43ms
2026-05-28T03:14:12.722Z INFO  [audit]        write event=login user=u_88310 ip=198.51.100.7 ua=android/12.0
2026-05-28T03:14:12.901Z DEBUG [cache-3a1]    redis.get key=sess:bbf miss=true ttl=0s
2026-05-28T03:14:13.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=419MB cpu=4.0%
2026-05-28T03:14:13.140Z INFO  [supervisor]   spawn child cmd="/opt/acme/bin/worker --id=22" cwd=/opt/acme env_count=87
2026-05-28T03:14:13.220Z INFO  [order-svc-12] POST /orders 200 21ms user=u_88311 amount=6.50
2026-05-28T03:14:13.401Z WARN  [tls]          certificate expiring soon cn=*.acme.io days_remaining=11
2026-05-28T03:14:13.500Z INFO  [supervisor]   worker-22 restarted pid=39112 took=3.4s
2026-05-28T03:14:13.612Z INFO  [k8s-probe]    pod worker-22 phase=Running readiness=true
2026-05-28T03:14:13.722Z INFO  [recom]         shard-rebalance done shards-moved=2 took=2.9s
2026-05-28T03:14:13.811Z INFO  [api-gw-7f9c]  upstream worker-22 marked UP, weight=0.2 (warm-up)
2026-05-28T03:14:13.902Z INFO  [search-svc]   query took=10ms hits=4 q="mechanical keyboard rgb"
2026-05-28T03:14:14.001Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:14.118Z INFO  [order-svc-12] POST /orders 200 26ms user=u_88301 amount=15.00
2026-05-28T03:14:14.244Z DEBUG [cache-3a1]    redis.set key=sess:ccc ttl=600s size=412B
2026-05-28T03:14:14.401Z INFO  [audit]        write event=order.created order=ORD-44888 user=u_88301
2026-05-28T03:14:14.512Z WARN  [s3-uploader] partial upload: file=invoice/INV-44888.pdf parts=3/5 retrying
2026-05-28T03:14:14.602Z INFO  [payment-svc]  circuit-breaker pay.acme.io state=HALF_OPEN probes=1
2026-05-28T03:14:14.701Z INFO  [payment-svc]  probe ok latency=121ms status=200
2026-05-28T03:14:14.812Z INFO  [s3-uploader] upload complete file=invoice/INV-44888.pdf parts=5/5 took=298ms
2026-05-28T03:14:14.901Z INFO  [payment-svc]  probe ok latency=98ms status=200
2026-05-28T03:14:15.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=420MB cpu=3.9%
2026-05-28T03:14:15.122Z INFO  [payment-svc]  circuit-breaker pay.acme.io state=CLOSED probes_ok=3/3 reset
2026-05-28T03:14:15.244Z INFO  [order-svc-09] POST /orders 200 17ms user=u_88322 amount=58.80
May 28 03:14:15 host-db-master postgres[2174]: LOG:  checkpoint complete: wrote 4188 buffers (10.2%); 0 WAL file(s) added, 0 removed, 0 recycled; write=12.108 s, sync=0.044 s, total=12.169 s
2026-05-28T03:14:15.401Z WARN  [k8s-probe]    node node-edge-03 conditions: MemoryPressure=false (recovered)
2026-05-28T03:14:15.512Z INFO  [recom]         model v17 served qps=3105 p99_ms=41
2026-05-28T03:14:15.611Z INFO  [audit]        write event=worker.restored worker=worker-22 took=5.4s
2026-05-28T03:14:15.722Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:15.811Z INFO  [order-svc-12] POST /orders 200 23ms user=u_88330 amount=42.10
2026-05-28T03:14:15.901Z DEBUG [cache-3a1]    redis.get key=sess:9ab miss=false ttl=480s
--- begin frontend / edge / cdn segment ---
2026-05-28T03:14:16.012Z INFO  [cdn-edge-hkg] HIT  https://static.acme.io/app.abc123.js status=200 bytes=84217 cache=HIT pop=hkg1 colo=HKG client_ip=203.0.113.55
2026-05-28T03:14:16.044Z INFO  [cdn-edge-hkg] MISS https://static.acme.io/chunk-44ad.js status=404 bytes=0   cache=MISS pop=hkg1 origin_status=404 client_ip=203.0.113.55 referer=https://shop.acme.io/p/yda-pro-13-2026
[browser/chrome 124] [Console] GET https://static.acme.io/chunk-44ad.js net::ERR_ABORTED 404 (Not Found)
[browser/chrome 124] [Console] ChunkLoadError: Loading chunk 44ad failed.
  (error: Error: Loading chunk 44ad failed.
    at HTMLScriptElement.l (https://static.acme.io/app.abc123.js:1:14082)
    at Object.next (https://static.acme.io/app.abc123.js:1:9011)
    at https://static.acme.io/app.abc123.js:1:9119)
  (missing: https://static.acme.io/chunk-44ad.js)
  trigger: page=https://shop.acme.io/p/yda-pro-13-2026 user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15"
2026-05-28T03:14:16.118Z INFO  [api-gw-7f9c]  GET /v1/products?cat=4 200 8ms
[browser/chrome 124] [Warning] [Violation] Forced reflow while executing JavaScript took 51ms
[browser/chrome 124] [Warning] ResizeObserver loop completed with undelivered notifications. (occurs 17x in last 200ms)
{"csp-report":{"document-uri":"https://shop.acme.io/p/yda-pro-13-2026","referrer":"","violated-directive":"script-src-elem","effective-directive":"script-src-elem","original-policy":"default-src 'self'; script-src 'self' https://*.acme.io; img-src 'self' data: https://*.acmecdn.net; report-uri /csp-report","disposition":"enforce","blocked-uri":"https://evil-tracker.example.cn/beacon.js","line-number":1,"column-number":1,"source-file":"https://shop.acme.io/p/yda-pro-13-2026","status-code":200,"script-sample":""}}
2026-05-28T03:14:16.260Z WARN  [waf-edge]     blocked rule="OWASP CRS 941100 XSS" client_ip=198.51.100.91 url=https://shop.acme.io/search?q=%3Cscript%3Ealert(1)%3C/script%3E action=block ja3=t13d1516h2_8daaf6152771_b186095e22b6
2026-05-28T03:14:16.330Z INFO  [cdn-edge-hkg] HIT  https://static.acme.io/app.abc123.css status=200 bytes=18234 cache=HIT pop=hkg1
<<<truncated raw stdin from a misbehaving sidecar>>> \x00\x00\x00\x10heartbeat\x00ack\x00 ... (binary noise, length=4096)
2026-05-28T03:14:16.401Z ERROR [frontend-bff] React hydration mismatch: server rendered "$199.00" client computed "$249.00"
  component: <ProductPrice sku="YDA-PRO-13-2026">
  Stack: at ProductPrice (https://static.acme.io/chunks/product.js:1:18840)
         at section (https://static.acme.io/chunks/product.js:1:18012)
         at ProductPage (https://static.acme.io/chunks/product.js:1:23104)
         at Suspense (https://static.acme.io/chunks/react-dom.production.min.js:21:4128)
  cause: stale ISR snapshot served while promo SPRING10 toggled to VIP20
  page: /p/yda-pro-13-2026  rev_id=2026052800917
2026-05-28T03:14:16.502Z WARN  [frontend-bff] cache-revalidate triggered key=page:/p/yda-pro-13-2026 reason=hydration-mismatch
May 28 03:14:16 host-edge-03 nginx[8120]: 198.51.100.7 - - [28/May/2026:03:14:16 +0000] "GET /static/sourcemaps/app.abc123.js.map HTTP/1.1" 404 153 "-" "Mozilla/5.0"
May 28 03:14:16 host-edge-03 nginx[8120]: 198.51.100.7 - - [28/May/2026:03:14:16 +0000] "GET /assets/logo-v3.svg HTTP/1.1" 200 4218 "-" "Mozilla/5.0"
2026-05-28T03:14:16.701Z INFO  [api-gw-7f9c]  heartbeat ok rss=425MB cpu=4.6%
2026-05-28T03:14:16.812Z DEBUG [cache-3a1]    redis.mget keys=12 hit=11 miss=1
2026-05-28T03:14:16.901Z INFO  [search-svc]   query took=11ms hits=58 q="耳机"
<html><head><title>500 Internal Server Error</title></head><body><center><h1>500 Internal Server Error</h1></center><hr><center>nginx/1.25.4</center></body></html>   --(sourced from upstream POST /v1/recommend captured by edge probe)--
2026-05-28T03:14:17.001Z ERROR [recom]         POST /v1/recommend 500 12ms user=u_88301 trace_id=8e4c1a7d cause="ConnectionResetError(104, 'Connection reset by peer')"
  Traceback (most recent call last):
    File "recom/server.py", line 211, in handle
      feats = self.feat_store.fetch(user_id)
    File "recom/feat_store.py", line 88, in fetch
      resp = self._sess.post(f'{self.url}/batch', json={'uid':user_id}, timeout=0.8)
  ConnectionResetError: [Errno 104] Connection reset by peer
  http_url: http://feat-store-internal.acme.local:7780/batch
  upstream_pod: feat-store-7d8c9b9b9c-xq2pl ip=10.42.4.219 zone=us-east-1b
2026-05-28T03:14:17.044Z WARN  [api-gw-7f9c]  upstream 502 from recom POST /v1/recommend latency=13ms client_ip=203.0.113.55 user=u_88301 -> served stale fallback
2026-05-28T03:14:17.118Z INFO  [order-svc-12] POST /orders 200 22ms user=u_88340 amount=199.00
2026-05-28T03:14:17.220Z INFO  [audit]        write event=feed.stale-served reason=recom-5xx user=u_88301
{"ts":"2026-05-28T03:14:17.330Z","lvl":"warn","svc":"sentry-relay","event":{"type":"transaction","transaction":"GET /p/[sku]","contexts":{"trace":{"trace_id":"4f1c9a2b...","span_id":"a1b2c3d4","status":"internal_error"}},"tags":{"release":"shop@2026.05.28-rc4","environment":"prod","runtime":"node:20.11"},"breadcrumbs":[{"cat":"navigation","msg":"/->/p/yda-pro-13-2026"},{"cat":"console","level":"error","msg":"ChunkLoadError: Loading chunk 44ad failed."},{"cat":"fetch","msg":"GET /v1/recommend 502"}],"truncated":true}}
2026-05-28T03:14:17.401Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:17.512Z INFO  [supervisor]   spawn child cmd="/opt/acme/bin/recom --shard=4" pid=39220
2026-05-28T03:14:17.611Z ERROR [worker-19]    PaymentError: refund failed: idem-key reused with mismatched body, key=refund:ORD-44801 prev_amount=199.00 new_amount=189.00 user=u_87990
  Traceback (most recent call last):
    File "refund/handler.py", line 47, in run
      receipt = client.refund(order_id=oid, amount=amt, idempotency_key=key)
    File "payments/client.py", line 191, in refund
      raise IdempotencyConflict(f'idem-key reused: {key}')
  payments.exceptions.IdempotencyConflict: idem-key reused: refund:ORD-44801
2026-05-28T03:14:17.701Z WARN  [audit]        write event=refund.conflict order=ORD-44801 prev=199.00 new=189.00 op=u_87990  -- requires manual reconciliation
[grafana-alert] firing: name="recom_5xx_rate" labels={service="recom", env="prod"} value=0.071 threshold=0.02 since="2026-05-28T03:14:17Z" runbook="https://wiki.acme.io/runbook/recom-5xx"
[slack-webhook] POST https://hooks.slack.com/services/T0000/B0000/REDACTED 200 142ms payload={"channel":"#prod-alerts","text":":fire: recom 5xx 7.1% (>2%)","attachments":[{"fields":[{"title":"trace","value":"4f1c9a2b"}]}]}
2026-05-28T03:14:17.812Z INFO  [api-gw-7f9c]  POST /v1/cart 200 12ms user=u_88340
2026-05-28T03:14:17.901Z DEBUG [cache-3a1]    redis.get key=sess:7ef miss=false ttl=80s
2026-05-28T03:14:18.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=427MB cpu=4.7%
2026-05-28T03:14:18.117Z INFO  [order-svc-09] POST /orders 200 19ms user=u_88345 amount=22.00
[browser/chrome 124] [Console] Mixed Content: The page at 'https://shop.acme.io/p/yda-pro-13-2026' was loaded over HTTPS, but requested an insecure XMLHttpRequest endpoint 'http://legacy-pixel.acme.cn/track'. This request has been blocked; the content must be served over HTTPS.
[browser/safari 17] [Console] [Error] Failed to load resource: The certificate for this server is invalid. You might be connecting to a server that is pretending to be "img-cdn.acme.cn" which could put your confidential information at risk. (asset/hero-banner-2026.jpg, line 0)
2026-05-28T03:14:18.244Z ERROR [edge-mtls]   handshake failed: x509: certificate has expired or is not yet valid: current time 2026-05-28T03:14:18Z is after 2026-05-25T00:00:00Z host=img-cdn.acme.cn client=cdn-edge-hkg
2026-05-28T03:14:18.330Z WARN  [tls]          rotate-now task triggered for cn=img-cdn.acme.cn expired_3d_ago=true (escalating: PagerDuty P2 page-id=PD-2026-0589)
2026-05-28T03:14:18.402Z INFO  [api-gw-7f9c]  GET /v1/products?cat=7 200 9ms
2026-05-28T03:14:18.512Z DEBUG [search-svc]   parsed query bm25_terms=[耳机,默认服购] filters={"price":[0,500]}
2026-05-28T03:14:18.601Z INFO  [order-svc-12] POST /orders 200 24ms user=u_88349 amount=88.00
2026-05-28T03:14:18.712Z WARN  [auth-svc]     jwt verification slow: kid=ks-2025-09 took=412ms (jwks fetch fallback)
2026-05-28T03:14:18.811Z ERROR [auth-svc]     jwks fetch failed: dial tcp: lookup auth.acme.io on 169.254.20.10:53: read udp 169.254.20.10:53: i/o timeout retry=2/3
2026-05-28T03:14:18.901Z INFO  [auth-svc]     jwks fetch ok from cache (stale=true age=311s) keys=4
2026-05-28T03:14:19.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=429MB cpu=4.9%
2026-05-28T03:14:19.118Z INFO  [audit]        write event=login user=u_88349 ip=192.0.2.18 ua=chrome/124.0
[prometheus/scrape] target=http://recom-7:9090/metrics state=DOWN err="context deadline exceeded" (15s) consecutive_failures=3 -> marking unhealthy
[prometheus/scrape] target=http://order-svc-12:9090/metrics state=UP scrape_duration=78ms samples=3144
2026-05-28T03:14:19.244Z WARN  [recom]         shard-rebalance still in progress moves=1/2 elapsed=8.6s expected=<5s
2026-05-28T03:14:19.330Z ERROR [recom]         shard-rebalance stalled shard=4 reason="primary candidate worker-19 over budget rss=5.2G/4G"
2026-05-28T03:14:19.401Z INFO  [supervisor]   admission decision: deny worker-19 promotion, fallback to worker-21
2026-05-28T03:14:19.512Z INFO  [recom]         shard-rebalance promote shard=4 -> worker-21
2026-05-28T03:14:19.611Z INFO  [recom]         shard-rebalance done shards-moved=2 took=11.2s (slow)
??base64?? eyJlbnYiOiJwcm9kIiwic3ZjIjoiYW5hbHl0aWNzIiwiYmF0Y2giOlsiZTEiLCJlMiIsImUzIiwiZTQiXX0=??end??
2026-05-28T03:14:19.812Z INFO  [search-svc]   query took=15ms hits=2 q="  " (empty after trim)
2026-05-28T03:14:19.901Z DEBUG [cache-3a1]    redis.get key=sess:newuser miss=true ttl=0s -> initialize
2026-05-28T03:14:20.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=430MB cpu=4.5%
2026-05-28T03:14:20.122Z WARN  [websocket]     conn closed code=1011 reason="internal server error" path=/ws/notify user=u_88301 dur=43s msgs_sent=7 msgs_recv=2
2026-05-28T03:14:20.244Z INFO  [websocket]     reconnect from u_88301 backoff=1s attempt=1
2026-05-28T03:14:20.330Z ERROR [api-gw-7f9c]  client TLS abort: tls: client offered only unsupported versions: [301 302] client_ip=185.220.101.34 ja3=00000000000000000000000000000000 (likely scanner)
2026-05-28T03:14:20.401Z WARN  [waf-edge]     rate-limit bucket exceeded ip=185.220.101.34 rule=ip-burst limit=120/min observed=482 action=block ttl=600s
2026-05-28T03:14:20.512Z INFO  [api-gw-7f9c]  POST /v1/cart 200 14ms user=u_88349
2026-05-28T03:14:20.611Z INFO  [order-svc-12] POST /orders 200 21ms user=u_88349 amount=12.50
2026-05-28T03:14:20.701Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:20.812Z DEBUG [search-svc]   parsed query bm25_terms=[键盘] filters={}
[browser/chrome 124] [Console] Uncaught (in promise) TypeError: Cannot read properties of undefined (reading 'price')
      at PriceTag (https://static.acme.io/chunks/cart.js:1:9211)
      at renderWithHooks (https://static.acme.io/chunks/react-dom.production.min.js:14:7332)
      at updateFunctionComponent (https://static.acme.io/chunks/react-dom.production.min.js:14:9818)
  caused by: backend returned items[2] without `price` field (cart=u_88349, order_draft=ORD-DRAFT-993)
  reported via window.onerror -> /csp-report (envelope_id=env-2026052803-7791)
2026-05-28T03:14:21.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=431MB cpu=4.6%
2026-05-28T03:14:21.118Z WARN  [cart-svc]     defensive: missing field items[2].price in draft ORD-DRAFT-993, defaulting to 0.00 (will be rejected at checkout)
2026-05-28T03:14:21.220Z ERROR [cart-svc]     checkout blocked: 1 item has price=0.00, order=ORD-DRAFT-993 user=u_88349 trace_id=ab12cd34
2026-05-28T03:14:21.330Z INFO  [audit]        write event=checkout.blocked order=ORD-DRAFT-993 reason=zero_price_item user=u_88349
2026-05-28T03:14:21.401Z INFO  [api-gw-7f9c]  GET /v1/products?cat=11 200 9ms
2026-05-28T03:14:21.512Z DEBUG [cache-3a1]    redis.set key=draft:ORD-DRAFT-993 ttl=1800s size=1208B
2026-05-28T03:14:21.611Z INFO  [recom]         model v17 served qps=3088 p99_ms=47
2026-05-28T03:14:21.722Z INFO  [order-svc-09] POST /orders 200 16ms user=u_88353 amount=6.00
2026-05-28T03:14:21.812Z WARN  [s3-uploader]  503 from s3 bucket=invoice-prod, signed-url=https://s3.amazonaws.com/invoice-prod/INV-44889.pdf?...(redacted) retry=1/5
2026-05-28T03:14:21.901Z INFO  [s3-uploader]  upload ok bucket=invoice-prod INV-44889.pdf parts=5/5 took=412ms retry=2
2026-05-28T03:14:22.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=429MB cpu=4.5%
========================== END EDGE/FRONTEND SEGMENT ============================
2026-05-28T03:14:22.122Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:22.244Z INFO  [order-svc-12] POST /orders 200 23ms user=u_88360 amount=15.00
2026-05-28T03:14:22.330Z DEBUG [cache-3a1]    redis.get key=sess:9ab miss=false ttl=440s
2026-05-28T03:14:22.401Z INFO  [search-svc]   query took=9ms hits=14 q="不响应的鼠标"
2026-05-28T03:14:22.512Z INFO  [api-gw-7f9c]  POST /v1/login 200 31ms user=u_88361
2026-05-28T03:14:22.611Z ERROR [serverless]    cold-start fn=invoice-render runtime=node20 init_ms=1820 (budget=400) -> emit hot-pool warm-up
2026-05-28T03:14:22.711Z WARN  [serverless]    fn=invoice-render concurrency=128 throttled=4 region=cn-shanghai
2026-05-28T03:14:22.812Z INFO  [api-gw-7f9c]  GET /v1/products?cat=2 200 8ms
2026-05-28T03:14:22.911Z INFO  [audit]        write event=invoice.requested order=ORD-44888 user=u_88301
curl --trace - http://feat-store-internal.acme.local:7780/batch  ## debug capture
==     0000: 50 4f 53 54 20 2f 62 61 74 63 68 20 48 54 54 50 POST /batch HTTP
==     0010: 2f 31 2e 31 0d 0a 48 6f 73 74 3a 20 66 65 61 74 /1.1..Host: feat
==  ... handshake stuck for 3s, peer RST
==     CONN-RESET errno=104
2026-05-28T03:14:23.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=430MB cpu=4.7%
2026-05-28T03:14:23.122Z ERROR [recom]         POST /v1/recommend 500 9ms cause=ConnectionResetError trace_id=ce9f8123 user=u_88361 (2nd in 6s)
2026-05-28T03:14:23.244Z WARN  [supervisor]    feat-store: 3 RST in 10s -> isolate pod feat-store-7d8c9b9b9c-xq2pl ip=10.42.4.219
2026-05-28T03:14:23.330Z INFO  [supervisor]    drain pod feat-store-7d8c9b9b9c-xq2pl grace=15s
2026-05-28T03:14:23.401Z INFO  [api-gw-7f9c]  GET /v1/health 200 1ms
2026-05-28T03:14:23.512Z INFO  [order-svc-12] POST /orders 200 22ms user=u_88370 amount=199.00
2026-05-28T03:14:23.611Z INFO  [recom]         feat-store fallback to local-cache hit_rate=0.91 (degraded)
2026-05-28T03:14:23.722Z INFO  [recom]         POST /v1/recommend 200 12ms fallback=local-cache user=u_88370
2026-05-28T03:14:23.812Z DEBUG [cache-3a1]    redis.get key=sess:abc miss=false ttl=210s
2026-05-28T03:14:23.901Z INFO  [search-svc]   query took=12ms hits=44 q="laptop bag"
2026-05-28T03:14:24.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=431MB cpu=4.5%
2026-05-28T03:14:24.118Z INFO  [supervisor]    drain complete pod=feat-store-7d8c9b9b9c-xq2pl, scheduling replacement
2026-05-28T03:14:24.244Z INFO  [supervisor]    spawn pod feat-store-7d8c9b9b9d-rx7nm ip=10.42.4.230
2026-05-28T03:14:24.330Z INFO  [k8s-probe]     readinessProbe pod=feat-store-7d8c9b9b9d-rx7nm statusCode=200 took=12ms
2026-05-28T03:14:24.401Z INFO  [recom]         feat-store endpoint refreshed, fallback off
2026-05-28T03:14:24.512Z INFO  [recom]         POST /v1/recommend 200 8ms user=u_88370 (recovered)
2026-05-28T03:14:24.611Z INFO  [audit]        write event=feat-store.replaced old=xq2pl new=rx7nm took=1.4s
2026-05-28T03:14:24.722Z WARN  [grafana-alert] resolved: name="recom_5xx_rate" labels={service="recom", env="prod"} value=0.004 since_resolve="2026-05-28T03:14:24Z" duration=7s
2026-05-28T03:14:24.812Z INFO  [api-gw-7f9c]  POST /v1/cart 200 11ms user=u_88370
2026-05-28T03:14:24.901Z DEBUG [cache-3a1]    redis.set key=sess:newuser ttl=600s size=256B
2026-05-28T03:14:25.001Z INFO  [api-gw-7f9c]  heartbeat ok rss=429MB cpu=4.4%
'''


# ──────────────────────────────────────────────────────────────────────
# 组装 prompts
# ──────────────────────────────────────────────────────────────────────
def build_prompts() -> List[Dict[str, Any]]:
    """构造五个场景的 Trajectory dict 列表。"""
    cases = [
        ('Python 代码', PY_QUERY, PY_PASSAGE),
        ('中文长篇新闻', NEWS_QUERY, NEWS_PASSAGE),
        ('网页 HTML', HTML_QUERY, HTML_PASSAGE),
        ('Python 异常处理', EXCEPTIONS_QUERY, EXCEPTIONS_PASSAGE),
        ('混合服务日志', LOGS_QUERY, LOGS_PASSAGE),
    ]
    prompts: List[Dict[str, Any]] = []
    for tag, query, passage in cases:
        # 50% 硬上限，与训练时一致
        budget = max(1, len(passage) // 2)
        user_msg = CONDENSER_USER.format(query=query, budget=budget, text=passage)
        prompts.append({
            'tag': tag,
            'src_len': len(passage),
            'budget': budget,
            'messages': [
                {'role': 'system', 'content': CONDENSER_SYSTEM},
                {'role': 'user', 'content': user_msg},
            ],
        })
    return prompts


def main():
    # 1. 初始化 Twinkle + Ray
    device_groups = [
        DeviceGroup(name='sampler',
                    ranks=list(range(SAMPLER_GPUS)),
                    device_type='GPU',
                    gpus_per_worker=SAMPLER_GPUS),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=SAMPLER_GPUS, groups=device_groups)

    # 2. 构造 vLLMSampler，max_model_len 需容纳 5120 字符级原文 + 系统提示 + 输出
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.7,
            'max_model_len': 32768,
            'enable_lora': False,
            'max_loras': 1,
            'max_lora_rank': 32,
            # 'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=32768)
    logger.info(get_device_placement())

    # 3. 采样参数：压缩任务用偏低温度，避免幻觉
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=0.1,
        top_p=0.9,
        num_samples=1,
    )

    # 4. 推理
    prompts = build_prompts()
    logger.info(f'共 {len(prompts)} 个压缩场景，模型 {MODEL_ID}，LoRA {LORA_PATH} ...')

    responses = sampler.sample(
        [{'messages': p['messages']} for p in prompts],
        sampling_params,
        # adapter_path=LORA_PATH,
    )

    # 5. 输出结果
    for i, response in enumerate(responses):
        meta = prompts[i]
        for seq in response.sequences:
            # strip chat-template close tag that leaks through decode
            text = seq.decoded.replace('<|im_end|>', '').rstrip()
            logger.info(
                f'\n{"=" * 60}\n'
                f'场景 {i + 1}：{meta["tag"]}（原文 {meta["src_len"]} 字符，硬上限 {meta["budget"]} 字符）\n'
                f'{"-" * 60}\n'
                f'压缩结果（{len(text)} 字符）：\n{text}\n')

    logger.info('全部场景压缩完成。')


if __name__ == '__main__':
    main()
