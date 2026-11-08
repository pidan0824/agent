"""Build retrieval queries and searchable text from facts and current Agent context."""

from collections import Counter, defaultdict
from statistics import median
from agent_simulation.data.common import require


def product_text(product):
    """Keep descriptions useful for retrieval; IDs and provenance stay in source records."""
    fields = (
        "item_title",
        "brand_name",
        "category_level1_name",
        "category_level2_name",
        "category_level3_name",
    )
    return "；".join(dict.fromkeys(str(product[k]) for k in fields if product.get(k) is not None))


def event_text(role, event, products=()):
    content = event["content"]
    if role == "customer":
        lines = ["搜索：" + content["query"]]
        for field, label in [("clicked_items", "点击"), ("purchased_items", "购买")]:
            if field not in content:
                lines.append(label + "：未提供")
                continue
            items = content[field]
            lines.append(
                label
                + "："
                + ("无记录" if not items else "\n".join(product_text(p) for p in items))
            )
        return "\n".join(lines)
    require(role == "seller", "Unknown retrieval role")
    by_id = {p["item_id"]: p for p in products}
    lines = []
    for row in content:
        require(row["item_id"] in by_id, "Seller retrieval requires the owned product description")
        price = "未知" if row["price_raw"] is None else str(row["price_raw"])
        lines.append(
            f"商品：{product_text(by_id[row['item_id']])}\n"
            f"价格原始值：{price}；价格状态：{row['price_status']}；"
            f"曝光记录：{row['exposure_records']}；点击记录：{row['click_records']}；"
            f"购买记录：{row['purchase_records']}"
        )
    return "\n".join(lines)


def seller_window_text(context, focal, products):
    """Describe the visible per-item windows, keeping the two clocks separate."""
    descriptions = {p["item_id"]: p for p in context.get("product_details", [])}
    windows = defaultdict(lambda: defaultdict(list))
    selected = set(focal)
    for event in context.get("short_term_memory", []):
        clock = "历史观测" if "time_index" in event else "仿真结算"
        for row in event["content"]:
            if row["item_id"] in selected:
                windows[clock][row["item_id"]].append(row)
    lines = []
    if len(focal) <= 5:
        for item in focal:
            lines.append("关注商品：" + product_text(descriptions.get(item, products[item])))
            for clock, items in windows.items():
                rows = items.get(item, [])
                if rows:
                    prices = [
                        str(r["price_raw"]) if r["price_raw"] is not None else "未知" for r in rows
                    ]
                    sequence = [
                        price for i, price in enumerate(prices) if not i or price != prices[i - 1]
                    ]
                    lines.append(
                        f"{clock}窗口：价格原始值按先后 {' → '.join(sequence)}；"
                        + counts_text(rows)
                    )
    else:
        lines.append(f"关注{len(focal)}件商品的经营经验和平台曝光假设")
        for clock, items in windows.items():
            changes = [
                b["price_raw"] - a["price_raw"]
                for rows in items.values()
                for a, b in zip(rows, rows[1:])
                if a["price_raw"] is not None and b["price_raw"] is not None
            ]
            lines.append(
                f"{clock}窗口：同商品相邻可比价格上涨{sum(d > 0 for d in changes)}次、"
                f"持平{changes.count(0)}次、下降{sum(d < 0 for d in changes)}次；"
                + counts_text([r for rows in items.values() for r in rows])
            )
    return lines


def counts_text(rows):
    exposures = [row["exposure_records"] for row in rows]
    clicks = sum(row["click_records"] for row in rows)
    purchases = sum(row["purchase_records"] for row in rows)
    scale = (
        "每个观测点仅有一次曝光"
        if all(n == 1 for n in exposures)
        else f"每个观测点平均有{sum(exposures) / len(rows):.2f}次曝光"
    )
    response = f"观察到{purchases}条购买记录" if purchases else "未观察到购买"
    return f"{len(rows)}个观测点共{sum(exposures)}条曝光记录，{scale}，点击记录{clicks}，{response}"


def query_text(role, context, latest=None):
    """Encode the task and focal information, never the entire shop or candidate list."""
    if isinstance(context, str):
        require(bool(context.strip()), "Empty memory query")
        return context
    require(type(context) is dict, "Memory query must be text or context")
    if latest is not None:
        products = context.get("products", [])
        task = "总结购物经历与消费偏好" if role == "customer" else "总结商品价格、曝光和成交经历"
        return task + "\n" + event_text(role, latest, products)
    if role == "customer":
        if context.get("query"):
            text = "查找购物经历与消费偏好。当前搜索：" + context["query"]
            offers = context.get("displayed_items", [])
            if offers:
                categories = Counter(p["category_level3_name"] for p in offers
                                     if p.get("category_level3_name") not in (None, "", "UNKNOWN"))
                prices = [p["price_raw"] for p in offers if (p.get("price_raw") or 0) > 0]
                if categories:
                    text += "；展示品类：" + "、".join(c for c, _ in categories.most_common(3))
                if prices:
                    text += f"；当前展示价（分）最低 {min(prices):g}，中位 {median(prices):g}，最高 {max(prices):g}"
            return text
        past = context.get("short_term_memory", {}).get("search_history", [])
        return (
            "回顾消费偏好，为下一次搜索提供参考。\n" + event_text(role, {"content": past[-1]})
            if past
            else "回顾消费偏好，为下一次搜索提供参考。"
        )
    require(role == "seller", "Unknown retrieval role")
    focal = context.get("history_presentation", {}).get(
        "detail_item_ids", context.get("requested_price_item_ids", [])
    )
    products = {p["item_id"]: p for p in context.get("owned_products", [])}
    require(set(focal) <= set(products), "Unknown focal product in memory query")
    task = "回顾价格、曝光、点击和购买之间的经营观察，为本店定价提供参考。"
    return "\n".join([task, *seller_window_text(context, focal, products)])
