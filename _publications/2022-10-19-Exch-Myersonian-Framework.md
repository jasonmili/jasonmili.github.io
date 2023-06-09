---
title: "A Myersonian Framework for Optimal Liquidity Provision in Automated Market Makers"
collection: publications
permalink: /publication/2022-10-19-Exch-Myersonian-Framework
date: 2022-10-19
venue: 'under submission'
excerpt: ''
paperurl: 'https://arxiv.org/pdf/2303.00208'
---
In decentralized finance ("DeFi"), automated market makers (AMMs) enable traders to programmatically exchange one asset for another. Such trades are enabled by the assets deposited by liquidity providers (LPs). The goal of this paper is to characterize and interpret the optimal (i.e., profit-maximizing) strategy of a monopolist liquidity provider, as a function of that LP's beliefs about asset prices and trader behavior.

We introduce a general framework for reasoning about AMMs. In this model, the market maker (i.e., LP) chooses a demand curve that specifies the quantity of a risky asset (such as BTC or ETH) to be held at each dollar price. Traders arrive sequentially and submit a price bid that can be interpreted as their estimate of the risky asset price; the AMM responds to this submitted bid with an allocation of the risky asset to the trader, a payment that the trader must pay, and a revised internal estimate for the true asset price. We define an incentive-compatible (IC) AMM as one in which a trader's optimal strategy is to submit its true estimate of the asset price, and characterize the IC AMMs as those with downward-sloping demand curves and payments defined by a formula familiar from Myerson's optimal auction theory. We characterize the profit-maximizing IC AMM via a generalization of Myerson's virtual values. The optimal demand curve generally has a jump that can be interpreted as a "bid-ask spread," which we show is caused by a combination of adverse selection risk (dominant when the degree of information asymmetry is large) and monopoly pricing (dominant when asymmetry is small).

[Download paper here](https://arxiv.org/pdf/2303.00208)
