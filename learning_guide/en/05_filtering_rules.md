# Chapter 5: Filtering Rules Explained

Recommendation systems must not only decide "what to recommend", but also "what **not** to recommend". Filtering is the defense line for guaranteeing user experience, complying with laws and regulations, and maintaining community atmosphere.

Code location: `home-mixer/filters/`

## 5.1 Two Stages of Filtering

In `PhoenixCandidatePipeline`, filtering happens in two stages:

1.  **Pre-Scoring Filters**:
    *   **Purpose**: Eliminate obviously unsuitable candidates as early as possible to reduce the number sent for model scoring, saving computational resources (GPUs are expensive).
    *   **Input**: Raw candidate set (thousands).
2.  **Post-Selection Filters**:
    *   **Purpose**: Perform final compliance checks on the selected Top-K.
    *   **Input**: Only dozens.

```mermaid
graph TD
    Start((Raw Candidates)) --> PreScore[Pre-Scoring Phase<br/>(Massive ~1000s)]
    
    subgraph "1. Pre-Scoring Filters"
        PreScore --> Dedup[DropDuplicates<br/>Dedup]
        Dedup --> Age[AgeFilter<br/>Timeliness]
        Age --> Self[SelfTweet<br/>Filter Self]
        Self --> Seen[PreviouslySeen<br/>Read Dedup]
        Seen --> Muted[MutedKeyword<br/>Keyword Muting]
        Muted --> Social[AuthorSocialgraph<br/>Social Relations]
        Social --> Sub[IneligibleSubscription<br/>Paywall Rights]
    end
    
    Sub --> Scoring[Scoring & Ranking<br/>Model Scoring & Sorting]
    Scoring --> TopK[Top-K Selection<br/>Truncation]
    
    TopK --> PostSel[Post-Selection Phase<br/>(Few ~Top 20)]
    
    subgraph "2. Post-Selection Filters"
        PostSel --> VFFilter[VF Filter<br/>Visibility Service Check]
        VFFilter --> DedupConv[DedupConversation<br/>Conversation Dedup]
    end
    
    DedupConv --> Final((Final Feed))
    
    style PreScore fill:#e1f5fe,stroke:#01579b
    style PostSel fill:#fff3e0,stroke:#ff6f00
    style VFFilter fill:#ffcdd2,stroke:#b71c1c
```

## 5.2 Core Filters Explained

### 5.2.1 Basic Quality Control
*   **`DropDuplicatesFilter`**: Dedup. Prevents the same tweet from appearing twice.
*   **`AgeFilter`**: Timeliness control. E.g., filter out tweets older than 48 hours (unless it's specially popular long-tail content).
*   **`CoreDataHydrationFilter`**: Data integrity check. If a tweet fails to even fetch text content, discard directly.

### 5.2.2 User Experience Protection
*   **`SelfTweetFilter`**: Do not recommend user's own tweets.
*   **`PreviouslySeenPostsFilter`**: Read dedup. Tweets seen by the user are not recommended again in a short time.
*   **`PreviouslyServedPostsFilter`**: Session dedup. Prevents tweets just seen from appearing again when user pulls to refresh.
*   **`RetweetDeduplicationFilter`**: Retweet dedup. If tweet A is a retweet of tweet B, and B is already in the candidate set, dedup.

### 5.2.3 Blocking & Blacklists
*   **`AuthorSocialgraphFilter`**: Social relation filtering.
    *   If user Blocks author -> Filter.
    *   If user Mutes author -> Filter.
*   **`MutedKeywordFilter`**: Keyword muting.
    *   Check if tweet text contains muted keywords set by the user. This is a text matching process.

### 5.2.4 Rights & Compliance
*   **`IneligibleSubscriptionFilter`**: Paywall filtering. If a tweet is Subscribers Only, and current user is not subscribed to the author, must filter.
*   **`VFFilter` (Visibility Filtering)**: Visibility filtering (usually in Post-Selection stage).
    *   This is the strictest review layer.
    *   Calls `VisibilityFilteringClient` service.
    *   Checks if content involves violence, pornography, hate speech, illegal content, etc.
    *   If a tweet is marked `nsfw` (Not Safe For Work) and user checks "Do not see sensitive content", it will also be filtered here.

## 5.3 Filter Implementation Pattern

All Filters implement the `Filter` trait:

```rust
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for MyFilter {
    async fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> Result<Vec<PostCandidate>, String> {
        // 1. Prepare data for filtering
        // 2. Iterate candidates
        // 3. Return kept candidates
    }
}
```

This pattern makes it very easy to verify filtering logic via unit tests, e.g., constructing a tweet containing muted keywords and asserting it is eliminated by `MutedKeywordFilter`.

---
**Summary**: We have fully walked through the entire process from retrieval to final display. Hope this guide helps you quickly get started with X's recommendation algorithm system!
