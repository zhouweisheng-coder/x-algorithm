# 第5章：过滤机制全解 (Filtering Rules)

推荐系统不仅要决定“推荐什么”，更要决定“**不**推荐什么”。过滤机制（Filtering）是保障用户体验、遵守法律法规和维护社区氛围的防线。

代码位置：`home-mixer/filters/`

## 5.1 过滤的两个阶段

在 `PhoenixCandidatePipeline` 中，过滤发生在两个阶段：

1.  **Pre-Scoring Filters (打分前过滤)**:
    *   **目的**: 尽早剔除明显不合适的候选，减少送给模型打分的数量，节省计算资源（GPU 是昂贵的）。
    *   **输入**: 原始候选集（几千条）。
2.  **Post-Selection Filters (选中后过滤)**:
    *   **目的**: 对最终选中的 Top-K 进行最后的合规性检查。
    *   **输入**: 只有几十条。

```mermaid
graph TD
    Start((原始候选集)) --> PreScore[Pre-Scoring Phase<br/>(海量候选 ~1000s)]
    
    subgraph "1. Pre-Scoring Filters (打分前过滤)"
        PreScore --> Dedup[DropDuplicates<br/>去重]
        Dedup --> Age[AgeFilter<br/>时效性]
        Age --> Self[SelfTweet<br/>过滤自己]
        Self --> Seen[PreviouslySeen<br/>已读去重]
        Seen --> Muted[MutedKeyword<br/>关键词屏蔽]
        Muted --> Social[AuthorSocialgraph<br/>社交关系]
        Social --> Sub[IneligibleSubscription<br/>付费墙权限]
    end
    
    Sub --> Scoring[Scoring & Ranking<br/>模型打分与排序]
    Scoring --> TopK[Top-K Selection<br/>截断]
    
    TopK --> PostSel[Post-Selection Phase<br/>(少量候选 ~Top 20)]
    
    subgraph "2. Post-Selection Filters (选中后过滤)"
        PostSel --> VFFilter[VF Filter<br/>可见性服务检测]
        VFFilter --> DedupConv[DedupConversation<br/>对话去重]
    end
    
    DedupConv --> Final((最终 Feed))
    
    style PreScore fill:#e1f5fe,stroke:#01579b
    style PostSel fill:#fff3e0,stroke:#ff6f00
    style VFFilter fill:#ffcdd2,stroke:#b71c1c
```

## 5.2 核心过滤器详解

### 5.2.1 基础质量控制
*   **`DropDuplicatesFilter`**: 去重。防止同一条推文出现两次。
*   **`AgeFilter`**: 时效性控制。例如，过滤掉 48 小时之前的推文（除非是特别热门的长尾内容）。
*   **`CoreDataHydrationFilter`**: 数据完整性检查。如果某条推文连文本内容都拉取失败了，直接丢弃。

### 5.2.2 用户体验保护
*   **`SelfTweetFilter`**: 不推荐用户自己发的推文。
*   **`PreviouslySeenPostsFilter`**: 已读去重。用户看过的推文，短时间内不再推荐。
*   **`PreviouslyServedPostsFilter`**: 本次会话去重。防止用户下拉刷新时，刚刷到的推文又出现。
*   **`RetweetDeduplicationFilter`**: 转发去重。如果推文 A 是推文 B 的转发，且 B 已经在候选集里了，去重。

### 5.2.3 屏蔽与黑名单
*   **`AuthorSocialgraphFilter`**: 社交关系过滤。
    *   如果用户 Block（拉黑）了作者 -> 过滤。
    *   如果用户 Mute（屏蔽）了作者 -> 过滤。
*   **`MutedKeywordFilter`**: 关键词屏蔽。
    *   检查推文文本是否包含用户设置的屏蔽词。这是一个文本匹配过程。

### 5.2.4 权限与合规
*   **`IneligibleSubscriptionFilter`**: 付费墙过滤。如果推文是仅订阅者可见（Subscribers Only），而当前用户没有订阅该作者，必须过滤。
*   **`VFFilter` (Visibility Filtering)**: 可见性过滤（通常在 Post-Selection 阶段）。
    *   这是最严格的审核层。
    *   调用 `VisibilityFilteringClient` 服务。
    *   检查内容是否涉及暴力、色情、仇恨言论、法律禁止内容等。
    *   如果推文被标记为 `nsfw` (Not Safe For Work) 且用户设置了“不看敏感内容”，也会在这里被过滤。

## 5.3 过滤器的实现模式

所有的 Filter 都实现了 `Filter` trait：

```rust
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for MyFilter {
    async fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> Result<Vec<PostCandidate>, String> {
        // 1. 准备过滤所需的数据
        // 2. 遍历 candidates
        // 3. 返回保留下来的 candidates
    }
}
```

这种模式使得我们可以非常容易地通过单元测试来验证过滤逻辑是否正确，例如构造一个包含屏蔽词的推文，断言它被 `MutedKeywordFilter` 剔除了。

---
**总结**: 至此，我们已经完整梳理了从召回到最终展示的全部流程。希望这份文档能帮助你快速上手 X 的推荐算法系统！