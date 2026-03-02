# User Engagement and Activity
A fundamental challenge in MoE implementation is the inherent class imbalance within user activity distributions. For a routing mechanism to develop specialized expertise, it must receive balanced exposure to different behavioral domains during the joint training phase.1 The data under review categorizes users into several tiers of activity, ranging from "Full active" to "Single low active," each presenting distinct challenges for a personalized ranking system.

### Distribution Dynamics and Bias
The analysis of user_active_degree distribution reveals that the majority of the population clusters within the "Full active" and "High active" segments. Specifically, "Full active" users account for nearly half of the interaction logs, posing a significant risk of routing collapse, in which the gating network defaults to experts optimized for high-frequency patterns.

| Engagement Level | Prevalence in Dataset | Behavioral Implications for Experts |
|------------------|----------------------:|-------------------------------------|
| Full active | 64.0% | High signal-to-noise ratio; requires experts specialized in deep personalization and exploit-heavy strategies. |
| High active | 23.3% | Consistent engagement requires experts capable of balancing exploration and exploitation. |
| Middle active | 8.3% | Critical transition segment; experts must focus on retention-driving content to prevent churn. |
| 2-14 day new | 2.3% | Cold-start regime; experts must prioritize high-entropy discovery to map nascent preference spaces. |
| Low active | 1.7% | Sparse signals; requires cross-domain or demographic-based routing to mitigate data poverty. |
| 30-day retention | < 0.2% | Extreme churn risk; requires survival-model-informed gating to maximize long-term reward. |

If the gating network is not regularized via auxiliary load-balancing losses or alternative strategies like "Loss-Free Balancing," the "Full active" majority will dominate the gradient updates.11 This dominance prevents the emergence of specialized "Newbie" experts, leading to sub-optimal performance for the very users most in need of accurate discovery—the recent registrants and low-activity participants.

# Distinguishing Creators from Consumers
A primary axis of differentiation in short-video ecosystems is the user's role in content production. The data demonstrate a strong correlation between creator status and overall platform activity, providing a high-confidence signal for expert routing.

### Live Streamer Status and Engagement
Users identified as live streamers exhibit fundamentally different engagement profiles than those of passive consumers. The cross-tabulation of live streaming status against engagement levels provides a clear signal for architectural partitioning.

| Is the User a Live Streamer? | Full active | High active | Middle active | Low active |
|------------------------------|------------:|------------:|--------------:|-----------:|
| Yes | 17.5% | 2.5% | 0.8% | 0.1% |
| No  | 46.5% | 20.8% | 7.5% | 1.6% |

The data indicates that while streamers comprise a smaller portion of the total user base, they are disproportionately represented in the "Full active" tier. From an MoE perspective, this suggests that the model should activate "Creator-Centric" experts when the user is a streamer, as their interaction patterns likely involve community management, social reciprocation, and high-frequency content browsing, which differ from the consumption-only patterns of the "No" group.

### Video Upload Status and Production Habits
The presence of uploaded videos (is_video_author) is an even stronger differentiator for activity levels. Users who have uploaded content are highly likely to be in the "Full active" segment (54.50%) compared to non-uploaders (9.5%).

| Has Uploaded Video? | Full active | High active | Middle active | Low active |
|---------------------|------------:|------------:|--------------:|-----------:|
| Yes | 54.5% | 16.3% | 6.2% | 1.5% |
| No  | 9.5%  | 7.0%  | 2.1% | 0.2% |

This split highlights a critical "Producer-Consumer" divide. In MoE systems, this justifies the use of "Shared Experts" to capture universal video-viewing preferences and "Specific Experts" to handle the social feedback loops unique to authors. The "Producer" experts might focus on signals like comment replies and shares, while "Consumer" experts optimize for watch time and discovery. 

# Social Capital and Network Topology

The metrics of "Social Capital"—fans, following, and friends—provide deep insights into a user's role within the community. These features represent the user’s "Embeddedness" and are essential for modeling social influence in the recommendation chain.

### Network Density and the Fan-Follower Relationship
The correlation between "Number of Following Users" and "Number of Fans" creates a clear spatial differentiation between passive consumers and influential creators. Visual analysis of this relationship reveals three distinct archetypes that are critical for expert routing:

1. The Passive Consumer (Lurker): Represented by the dense cluster near the origin. These users follow very few people and have almost zero fans. They act primarily as "Passive Spectators" who consume content without participating in the social network or production.

2. The Social Thriver: Users who follow a high number of others (approaching the 5,000 cap) but maintain a moderate fan count (10k–20k). Their behavior is characterized by high reciprocity and community-driven interaction.

3. The Mega-Influencer: Exceptional outliers who follow very few users (often < 100) yet command extreme fan counts (up to 160,000). These are professional content producers where the "Brand Maturity" is at its peak.

The following counts serve as proxies for the user’s social appetite. The EDA shows that users with a following count of "500+" are almost entirely "Full active."

| Following Range | Full active | High active | Middle active | Low active |
|-----------------|------------:|------------:|--------------:|-----------:|
|        0 |  0.6% | 0.2% | 0.1% | 0.0% |
|   (0, 10]|  3.3% | 2.1% | 0.4% | 0.2% |
| (50, 100]|  8.5% | 4.1% | 0.5% | 0.1% |
|(250, 500]| 12.1% | 3.0% | 1.4% | 0.6% |
|      500+| 14.0% | 3.9% | 1.6% | 0.4% |

In an MoE model, the gating network should use the "Fan-Follower Ratio" to route "Social Thrivers" to experts that prioritize community-driven content and "Electronic Word-of-Mouth" signals, while routing influencers to experts that mitigate popularity bias.

### Influence Hierarchy: Fan Distribution
While "Following" represents active interest, "Fans" represent passive influence or celebrity. The EDA heatmap for "Number of Fans" reveals a "Long Tail" distribution.

| Fan Range | Full active | High active | Middle active | Implications |
|-----------|------------:|------------:|--------------:|--------------|
|         0 | 1.5%  | 1.8% | 1.1% | Lurker / Newbie profile. |
|   [1, 10) | 11.3% | 7.9% | 2.1% | Micro-influence; high engagement. |
| [10, 100) | 24.7% | 8.7% | 2.9% | Peak community interaction. |
| [100, 1k) | 21.8% | 4.2% | 2.2% | Aspiring creators: high signal. |
|  [1k, 5k) | 3.2%  | 0.6% | 0.0% | Professional content producers. |

The MoE router should distinguish between these "Micro-Influencers" and "Mega-Influencers," as the latter require experts that account for "Popularity Bias" and "Celebrity Context" rather than peer-to-peer relatability.

# Account Maturity

The "Registration Age" is perhaps the most stable predictor of user behavior, representing the transition from a "Cold-Start" exploratory phase to a "Veteran" exploitation phase.

### The Veteran Dominance
The analysis reveals a profound concentration of engagement among long-term users. More than half (52.00%) of all interactions come from "Full active" users who have been registered for over 730 days.

| Registration Days | Full active | High active | Middle active | Single low active |
|-------------------|------------:|------------:|--------------:|------------------:|
| 15-30 | 0.2% | 0.1% | 0.0% | 0.0% |
| 91-180 | 1.5% | 1.5% | 0.7% | 0.0% |
| 366-730 | 5.8% | 3.5% | 1.2% | 0.2% |
| 730+ | 52.0% | 14.8% | 4.7% | 0.0% |

This distribution suggests that "Veteran" users are the primary beneficiaries of the system's current recommendation logic. The model needs "Veteran Experts" that can navigate niche interests, but it also needs "Newbie Experts" that are not drowned out by the veteran signal.

### Trajectory Analysis: Fans vs. Time
The relationship between time on the platform and fan growth further differentiates user types. Scatter plots show that for "Non-creators," fan growth is virtually non-existent regardless of registration age. In contrast, "Creators" show a distinct, albeit noisy, upward trajectory in fan acquisition over time. This suggests that for "Video Authors," the registration age is a proxy for "Brand Maturity." The MoE system should route these authors to different experts: one specialized in "Maintaining Loyalty" and another in "Aggressive Exposure and Growth.

# Dimensionality Reduction

The transition from raw EDA to model input requires careful feature transformation to maximize the "Differentiability" of users.

### Dimensionality Reduction via Autoencoders
To manage high-dimensional user feature sets, incorporating an autoencoder-based preprocessing layer is highly effective. Autoencoders utilize non-linear functions to capture the complex dependencies inherent in consumption patterns.

- Latent Space Embeddings: Deploy a Deep Autoencoder (DAE) to compress sparse features into a dense latent space. This "bottleneck" representation ensures the gating network learns salient features while discarding noise.

- Variational Autoencoders (VAE): By modeling user preferences as probabilistic distributions, VAEs capture uncertainty in interests, which helps improve recommendation diversity.

- Denoising Autoencoders (DAE): Useful for reconstructing "true" user signals from corrupted or sparse inputs, leading to robust expert specialization.
Numerical vs. Categorical Representations

The recommendation is to use raw numerical values (normalized 0-1) for the gating network. Neural networks learn continuous relationships—such as the trajectory of fan growth against registration age—more effectively than binned categories.


# Mitigating Imbalance and Bias in Optimization

The dominance of veterans and full active users means that, without intervention, the engine will fail to attract or retain new users.

### Load Balancing Strategies
Because 90% of users may be "Full active," the gating network might learn to route every interaction to a single "Heavy Hitter" expert.

1. Capacity Factors: Limits the tokens any single expert can handle, forcing the router to find the "Second-Best" expert.

2. Expert Dropout: Randomly deactivates experts to prevent the gating network from becoming overly dependent on a single subnetwork.

3. Counterfactual Watch Time: Prevents duration-based bias by training experts to predict "Interest Level" rather than raw play duration.