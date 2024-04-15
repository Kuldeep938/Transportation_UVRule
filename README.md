# Transportation Problem Solver

## Overview

This repository contains an implementation of the North-West (NW) Rule, Minimum Cost (MinCost) Rule, and UV Rule for solving transportation problems. These rules are essential techniques used in optimizing transportation and logistics operations

## Features

- **North-West (NW) Rule:** This algorithm is used to find the initial basic feasible solution for a transportation problem. It starts by allocating shipments in the top-left corner of the cost matrix and iteratively adjusting the allocations based on supply and demand constraints.

- **Minimum Cost (MinCost) Rule:** After obtaining the initial basic feasible solution, the Minimum Cost Rule helps in finding the optimal solution by minimizing the total transportation cost. It identifies the most cost-effective routes for transporting goods.

- **UV Rule:** The UV Rule is a method for checking the optimality of the initial basic feasible solution. It helps in detecting whether the current solution is optimal or needs further improvement. If the solution is not optimal, the UV Rule provides guidance on how to adjust the allocations to improve cost efficiency.

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Kuldeep938/Transportation_UVRule.git
