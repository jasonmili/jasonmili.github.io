---
title: "Complexity-Approximation Trade-offs in Exchange Mechanisms: AMMs vs. LOBs"
collection: publications
permalink: /publication/2022-09-26-Exch-Approx-Complex-Tradeoffs
date: 2022-09-26
venue: 'FC 2023'
excerpt: ''
paperurl: 'https://link.springer.com/chapter/10.1007/978-3-031-47754-6_19'
---
This paper presents a general framework for the design and analysis of exchange mechanisms between two assets that unifies and enables comparisons between the two dominant paradigms for exchange, constant function market markers (CFMMs) and limit order books (LOBs). In our framework, each liquidity provider (LP) submits to the exchange a downward-sloping demand curve, specifying the quantity of the risky asset it wishes to hold at each price; the exchange buys and sells the risky asset so as to satisfy the aggregate submitted demand. In general, such a mechanism is budget-balanced and enables price discovery. Different exchange mechanisms correspond to different restrictions on the set of acceptable demand curves. The primary goal of this paper is to formalize an approximation-complexity trade-off that pervades the design of exchange mechanisms. For example, CFMMs give up expressiveness in favor of simplicity: the aggregate demand curve of the LPs can be described using constant space, but most demand curves cannot be well approximated by any function in the corresponding single-dimensional family. LOBs, intuitively, make the opposite trade-off: any downward-slowing demand curve can be well approximated by a collection of limit orders, but the space needed to describe the state of a LOB can be large. This paper introduces a general measure of {\em exchange complexity}, defined by the minimal set of basis functions that generate, through their conical hull, all of the demand functions allowed by an exchange. With this complexity measure in place, we investigate the design of {\em optimally expressive} exchange mechanisms, meaning the lowest complexity mechanisms that allow for arbitrary downward-sloping demand curves to be well approximated. As a case study, we interpret the complexity-approximation trade-offs in the widely-used Uniswap v3 AMM through the lens of our framework.

[Download paper here](https://link.springer.com/chapter/10.1007/978-3-031-47754-6_19)
