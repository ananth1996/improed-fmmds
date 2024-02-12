import functools
import sys
import time
from typing import Any, Callable, List, Union

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB
from tqdm.autonotebook import tqdm
import fmmd_old.utils as utils
import matplotlib.pyplot as plt
ElemList = Union[List[utils.Elem], List[utils.ElemSparse]]
TIME_LIMIT_ILP = 300


def get_initial_greedy_solution(V: ElemList,k,dist:Callable,_tqdm=False):
    # The Gonzalez's algorithm
    cand = set()
    cand_dists = dict()
    cand_div = sys.float_info.max
    array_dists = [sys.float_info.max] * len(V)

    # find greedy solution without fairness constraints

    # pick first element to start
    cand.add(0)
    cand_dists[0] = dict()
    # compute all distances to that element
    for i in range(len(V)):
        array_dists[i] = dist(V[0], V[i])
    # Round at 12 places to avoid issues with numerical stability
    array_dists = np.round(array_dists,12)
    # greedily pick k-1 other elements
    n = len(V)
    if _tqdm: pbar= tqdm(total=k-1)
    while len(cand) < k:
        max_idx = np.argmax(array_dists)
        max_dist = np.max(array_dists)
        cand.add(max_idx)
        # cand_dists[max_idx] = dict()
        # for idx in cand:
        #     if idx < max_idx:
        #         cand_dists[idx][max_idx] = dist(V[idx], V[max_idx])
        #     elif idx > max_idx:
        #         cand_dists[max_idx][idx] = dist(V[idx], V[max_idx])
        cand_div = min(cand_div, max_dist)
        # update the closest distance to current candidate set
        for i in range(n):
            array_dists[i] = min(array_dists[i], dist(V[i], V[max_idx]))
        # Round at 12 places to avoid issues with numerical stability
        array_dists = np.round(array_dists,12)
        if _tqdm: pbar.update()
    
    return cand,cand_div


def get_group_greedy_solution(V,k,C,constr,dist,eps):
    # Initialization
    sol = list()
    div_sol = 0.0

    # The Gonzalez's algorithm
    cand = set()
    cand_dists = dict()
    cand_div = sys.float_info.max
    array_dists = [sys.float_info.max] * len(V)

    # find greedy solution without fairness constraints

    # pick first element to start
    cand.add(0)
    cand_dists[0] = dict()
    # compute all distances to that element
    for i in range(len(V)):
        array_dists[i] = dist(V[0], V[i])
    # Round at 12 places to avoid issues with numerical stability
    array_dists = np.round(array_dists,10)
    # greedily pick k-1 other elements
    n = len(V)
    while len(cand) < k:
        max_idx = np.argmax(array_dists)
        max_dist = np.max(array_dists)
        cand.add(max_idx)
        cand_div = min(cand_div, max_dist)
        # update the closest distance to current candidate set
        for i in range(n):
            array_dists[i] = min(array_dists[i], dist(V[i], V[max_idx]))
        # Round at 12 places to avoid issues with numerical stability
        array_dists = np.round(array_dists,10)

    # Divide candidates by colors
    cand_colors = list()
    for c in range(C):
        cand_colors.append(set())
    for idx in cand:
        c = V[idx].color
        cand_colors[c].add(idx)

    # Compute the solution
    div = cand_div
    while True:
        under_capped = False
        for c in range(C):
            # Add an arbitrary element of color c when there is not anyone in the candidate.
            if len(cand_colors[c]) == 0:
                for i in range(len(V)):
                    if V[i].color == c:
                        cand_colors[c].add(i)
                        cand.add(i)
                        break
            # The Gonzalez's algorithm starting from cand_colors[c] on all elements of color c
            array_dists_color = [sys.float_info.max] * len(V)
            for i in range(len(V)):
                for j in cand_colors[c]:
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[j]))
                    else:
                        array_dists_color[i] = 0.0
            # Round at 10 places to avoid issues with numerical stability
            array_dists_color = np.round(array_dists_color,10)
            max_idx_c = np.argmax(array_dists_color)
            max_dist_c = np.max(array_dists_color)
            # print(f"group={c}")
            # print(f"Initial solution={cand_colors[c]}")
            # print(f"{div=}")
            # print(f"\t\t{max_dist_c=}")
            while len(cand_colors[c]) < k and max_dist_c >= div:
                cand_colors[c].add(max_idx_c)
                cand.add(max_idx_c)
                # print(f"\t\tAdding item {max_idx_c}")
                for i in range(len(V)):
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[max_idx_c]))
                    # Round at 10 places to avoid issues with numerical stability
                    array_dists_color = np.round(array_dists_color,10)
                max_idx_c = np.argmax(array_dists_color)
                max_dist_c = np.max(array_dists_color)
                # print(f"\t\t{max_dist_c=}")
            if len(cand_colors[c]) < constr[c][0]:
                under_capped = True
                div = div*(1-eps)
                print(f"Undercapped decreasing diversity to {div}")
                break
        if not under_capped:
            return cand,div



def fmmd(V: ElemList, EPS: float, k: int, C: int, constr: List[List[int]], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    # Initialization
    sol = list()
    div_sol = 0.0

    # The Gonzalez's algorithm
    cand = set()
    cand_dists = dict()
    cand_div = sys.float_info.max
    array_dists = [sys.float_info.max] * len(V)

    # find greedy solution without fairness constraints

    # pick first element to start
    cand.add(0)
    cand_dists[0] = dict()
    # compute all distances to that element
    for i in range(len(V)):
        array_dists[i] = dist(V[0], V[i])
    # Round at 10 places to avoid issues with numerical stability
    array_dists = np.round(array_dists,10)
    # greedily pick k-1 other elements
    n = len(V)
    while len(cand) < k:
        max_idx = np.argmax(array_dists)
        max_dist = np.max(array_dists)
        cand.add(max_idx)
        cand_dists[max_idx] = dict()
        for idx in cand:
            if idx < max_idx:
                cand_dists[idx][max_idx] = dist(V[idx], V[max_idx])
            elif idx > max_idx:
                cand_dists[max_idx][idx] = dist(V[idx], V[max_idx])
        cand_div = min(cand_div, max_dist)
        # update the closest distance to current candidate set
        for i in range(n):
            array_dists[i] = min(array_dists[i], dist(V[i], V[max_idx]))
        # Round at 10 places to avoid issues with numerical stability
        array_dists = np.round(array_dists,10)
        
    # print("Inital candidate set found")

    # Divide candidates by colors
    cand_colors = list()
    for c in range(C):
        cand_colors.append(set())
    for idx in cand:
        c = V[idx].color
        cand_colors[c].add(idx)

    # Compute the solution
    div = cand_div
    while len(sol) == 0:
        under_capped = False
        for c in range(C):
            # Add an arbitrary element of color c when there is not anyone in the candidate.
            if len(cand_colors[c]) == 0:
                for i in range(len(V)):
                    if V[i].color == c:
                        cand_colors[c].add(i)
                        cand.add(i)
                        cand_dists[i] = dict()
                        for idx in cand:
                            if idx < i:
                                cand_dists[idx][i] = dist(V[idx], V[i])
                            elif idx > i:
                                cand_dists[i][idx] = dist(V[idx], V[i])
                        break
            # The Gonzalez's algorithm starting from cand_colors[c] on all elements of color c
            array_dists_color = [sys.float_info.max] * len(V)
            for i in range(len(V)):
                for j in cand_colors[c]:
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[j]))
                    else:
                        array_dists_color[i] = 0.0
            # Round at 10 places to avoid issues with numerical stability
            array_dists_color = np.round(array_dists_color,10)
            max_idx_c = np.argmax(array_dists_color)
            max_dist_c = np.max(array_dists_color)
            while len(cand_colors[c]) < k and max_dist_c >= div:
                cand_colors[c].add(max_idx_c)
                cand.add(max_idx_c)
                cand_dists[max_idx_c] = dict()
                for idx in cand:
                    if idx < max_idx_c:
                        cand_dists[idx][max_idx_c] = dist(V[idx], V[max_idx_c])
                    elif idx > max_idx_c:
                        cand_dists[max_idx_c][idx] = dist(V[idx], V[max_idx_c])
                for i in range(len(V)):
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[max_idx_c]))
                # Round at 10 places to avoid issues w  ith numerical stability
                array_dists_color = np.round(array_dists_color,10)
                max_idx_c = np.argmax(array_dists_color)
                max_dist_c = np.max(array_dists_color)
            if len(cand_colors[c]) < constr[c][0]:
                under_capped = True
                break
        if under_capped:
            div = div * (1.0 - EPS)
            print(f"Undercapped decreasing diversity to {div}")
            continue
        
        # print("Coreset Solution Found")
        # Build a graph G w.r.t. cand_div
        # print("solution is ",sorted({V[i].idx for i in cand}))
        dict_cand = dict()
        new_idx = 0
        _mapping = {V[i].idx:i for i in cand}
        # add nodes from sorted true index values 
        for idx in sorted(list(_mapping.keys())):
            dict_cand[_mapping[idx]] = new_idx
            new_idx += 1
        dict_cand_r = {v:k for k,v in dict_cand.items()}
        G = nx.Graph()
        # ensure adding edges in node order
        G.add_nodes_from(range(len(cand)))
        for _i in range(len(cand)):
            for _j in range(_i+1,len(cand)):
                # map node id to idx
                i = dict_cand_r[_i]
                j = dict_cand_r[_j]
                if i < j:
                    d = cand_dists[i][j]
                else:
                    d = cand_dists[j][i]
                if np.round(d,10) < np.round(div,10):
                    G.add_edge(_i, _j)
        
        # print("Graph Built")
        # fig,ax = plt.subplots(figsize=(10,10))
        print(f"{G.number_of_edges()=} {G.number_of_nodes()=}")
        # pos = nx.circular_layout(G)
        # node_colors = [V[dict_cand_r[n]].color for n in G]
        # node_labels = {n:f"{n}\n{V[dict_cand_r[n]].idx}" for n in G.nodes()}
        # nx.draw_networkx_nodes(G,pos=pos,alpha=0.5,node_color=node_colors)
        # nx.draw_networkx_edges(G,pos=pos)
        # nx.draw_networkx_labels(G,pos=pos,labels=node_labels)
        # plt.show()

        # Find an independent set S of G using ILP
        try:
            model = gp.Model("mis_" + str(div))
            model.setParam(GRB.Param.OutputFlag, False)
            model.setParam(GRB.Param.TimeLimit, TIME_LIMIT_ILP)

            size = [1] * len(cand)
            vars_x = model.addVars(len(cand), vtype=GRB.BINARY, obj=size, name="x")

            model.modelSense = GRB.MAXIMIZE
            #! For Gurobi to be deterministic all constraints must 
            #! be in the same order. Ensure Graph node and edge order
            #! are consistent when testing
            # https://support.gurobi.com/hc/en-us/community/posts/360048065332-Order-of-variables-constraints-vs-performance
            eid = 0
            for e in G.edges:
                model.addConstr(vars_x[e[0]] + vars_x[e[1]] <= 1, "edge_" + str(eid))
                eid += 1

            expr = gp.LinExpr()
            for j in range(len(cand)):
                expr.addTerms(1, vars_x[j])
            model.addConstr(expr <= k, "size")

            for c in range(C):
                expr = gp.LinExpr()
                for j in cand_colors[c]:
                    expr.addTerms(1, vars_x[dict_cand[j]])
                model.addConstr(expr >= constr[c][0], "lb_color_" + str(c))
                model.addConstr(expr <= constr[c][1], "ub_color_" + str(c))
            model.optimize()
            # prints constraints
            # for constrs in model.getConstrs():
            #     print(constrs)
            #     lhs= model.getRow(constrs)
            #     rhs =model.getAttr("RHS",[constrs])
            #     sense = model.getAttr("Sense",[constrs])
            #     print(lhs,sense,rhs)
            #     print()
            
            S = set()
            for j in range(len(cand)):
                if vars_x[j].X > 0.5:
                    S.add(j)

            if len(S) >= k:
                for key, value in dict_cand.items():
                    if value in S:
                        sol.append(key)
                div_sol = utils.div_subset(V, sol, dist)
                break
            else:
                div = div * (1.0 - EPS)

        except gp.GurobiError as error:
            print("Error code " + str(error))
            exit(0)

        except AttributeError:
            div = div * (1.0 - EPS)
            print(f"ILP Error. Decreasing diversity threshold to {div}")

    return sol, div_sol



def fmmd_coresets(V: ElemList, EPS: float, k: int, C: int, constr: List[List[int]], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    # Initialization
    graphs = []
    sol = list()
    div_sol = 0.0

    # The Gonzalez's algorithm
    cand = set()
    cand_dists = dict()
    cand_div = sys.float_info.max
    array_dists = [sys.float_info.max] * len(V)

    # find greedy solution without fairness constraints

    # pick first element to start
    cand.add(0)
    cand_dists[0] = dict()
    # compute all distances to that element
    for i in range(len(V)):
        array_dists[i] = dist(V[0], V[i])
    # Round at 10 places to avoid issues with numerical stability
    array_dists = np.round(array_dists,10)
    # greedily pick k-1 other elements
    n = len(V)
    while len(cand) < k:
        max_idx = np.argmax(array_dists)
        max_dist = np.max(array_dists)
        cand.add(max_idx)
        cand_dists[max_idx] = dict()
        for idx in cand:
            if idx < max_idx:
                cand_dists[idx][max_idx] = dist(V[idx], V[max_idx])
            elif idx > max_idx:
                cand_dists[max_idx][idx] = dist(V[idx], V[max_idx])
        cand_div = min(cand_div, max_dist)
        # update the closest distance to current candidate set
        for i in range(n):
            array_dists[i] = min(array_dists[i], dist(V[i], V[max_idx]))
        # Round at 10 places to avoid issues with numerical stability
        array_dists = np.round(array_dists,10)
        
    # print("Inital candidate set found")

    # Divide candidates by colors
    cand_colors = list()
    for c in range(C):
        cand_colors.append(set())
    for idx in cand:
        c = V[idx].color
        cand_colors[c].add(idx)

    # Compute the solution
    div = cand_div
    while len(sol) == 0:
        under_capped = False
        for c in range(C):
            # Add an arbitrary element of color c when there is not anyone in the candidate.
            if len(cand_colors[c]) == 0:
                for i in range(len(V)):
                    if V[i].color == c:
                        cand_colors[c].add(i)
                        cand.add(i)
                        cand_dists[i] = dict()
                        for idx in cand:
                            if idx < i:
                                cand_dists[idx][i] = dist(V[idx], V[i])
                            elif idx > i:
                                cand_dists[i][idx] = dist(V[idx], V[i])
                        break
            # The Gonzalez's algorithm starting from cand_colors[c] on all elements of color c
            array_dists_color = [sys.float_info.max] * len(V)
            for i in range(len(V)):
                for j in cand_colors[c]:
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[j]))
                    else:
                        array_dists_color[i] = 0.0
            # Round at 10 places to avoid issues with numerical stability
            array_dists_color = np.round(array_dists_color,10)
            max_idx_c = np.argmax(array_dists_color)
            max_dist_c = np.max(array_dists_color)
            while len(cand_colors[c]) < k and max_dist_c >= div:
                cand_colors[c].add(max_idx_c)
                cand.add(max_idx_c)
                cand_dists[max_idx_c] = dict()
                for idx in cand:
                    if idx < max_idx_c:
                        cand_dists[idx][max_idx_c] = dist(V[idx], V[max_idx_c])
                    elif idx > max_idx_c:
                        cand_dists[max_idx_c][idx] = dist(V[idx], V[max_idx_c])
                for i in range(len(V)):
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[max_idx_c]))
                # Round at 10 places to avoid issues w  ith numerical stability
                array_dists_color = np.round(array_dists_color,10)
                max_idx_c = np.argmax(array_dists_color)
                max_dist_c = np.max(array_dists_color)
            if len(cand_colors[c]) < constr[c][0]:
                under_capped = True
                break
        if under_capped:
            div = div * (1.0 - EPS)
            print(f"Undercapped decreasing diversity to {div}")
            continue
        
        # print("Coreset Solution Found")
        # Build a graph G w.r.t. cand_div
        # print("solution is ",sorted({V[i].idx for i in cand}))
        dict_cand = dict()
        new_idx = 0
        _mapping = {V[i].idx:i for i in cand}
        # add nodes from sorted true index values 
        for idx in sorted(list(_mapping.keys())):
            dict_cand[_mapping[idx]] = new_idx
            new_idx += 1
        dict_cand_r = {v:k for k,v in dict_cand.items()}
        groups = {k:V[v].color for k,v in dict_cand_r.items()}
        ids = {k:V[v].idx for k,v in dict_cand_r.items()}
        G = nx.Graph()
        # ensure adding edges in node order
        G.add_nodes_from(range(len(cand)))
        nx.set_node_attributes(G,groups,"group")
        nx.set_node_attributes(G,ids,"id")
        for _i in range(len(cand)):
            for _j in range(_i+1,len(cand)):
                # map node id to idx
                i = dict_cand_r[_i]
                j = dict_cand_r[_j]
                if i < j:
                    d = cand_dists[i][j]
                else:
                    d = cand_dists[j][i]
                if np.round(d,10) < np.round(div,10):
                    G.add_edge(_i, _j)
        
        graphs.append(G.copy())
        # print("Graph Built")
        # fig,ax = plt.subplots(figsize=(10,10))
        print(f"{G.number_of_edges()=} {G.number_of_nodes()=}")
        # pos = nx.circular_layout(G)
        # node_colors = [V[dict_cand_r[n]].color for n in G]
        # node_labels = {n:f"{n}\n{V[dict_cand_r[n]].idx}" for n in G.nodes()}
        # nx.draw_networkx_nodes(G,pos=pos,alpha=0.5,node_color=node_colors)
        # nx.draw_networkx_edges(G,pos=pos)
        # nx.draw_networkx_labels(G,pos=pos,labels=node_labels)
        # plt.show()

        # Find an independent set S of G using ILP
        try:
            model = gp.Model("mis_" + str(div))
            model.setParam(GRB.Param.OutputFlag, False)
            model.setParam(GRB.Param.TimeLimit, TIME_LIMIT_ILP)

            size = [1] * len(cand)
            vars_x = model.addVars(len(cand), vtype=GRB.BINARY, obj=size, name="x")

            model.modelSense = GRB.MAXIMIZE
            #! For Gurobi to be deterministic all constraints must 
            #! be in the same order. Ensure Graph node and edge order
            #! are consistent when testing
            # https://support.gurobi.com/hc/en-us/community/posts/360048065332-Order-of-variables-constraints-vs-performance
            eid = 0
            for e in G.edges:
                model.addConstr(vars_x[e[0]] + vars_x[e[1]] <= 1, "edge_" + str(eid))
                eid += 1

            expr = gp.LinExpr()
            for j in range(len(cand)):
                expr.addTerms(1, vars_x[j])
            model.addConstr(expr <= k, "size")

            for c in range(C):
                expr = gp.LinExpr()
                for j in cand_colors[c]:
                    expr.addTerms(1, vars_x[dict_cand[j]])
                model.addConstr(expr >= constr[c][0], "lb_color_" + str(c))
                model.addConstr(expr <= constr[c][1], "ub_color_" + str(c))
            model.optimize()
            # prints constraints
            # for constrs in model.getConstrs():
            #     print(constrs)
            #     lhs= model.getRow(constrs)
            #     rhs =model.getAttr("RHS",[constrs])
            #     sense = model.getAttr("Sense",[constrs])
            #     print(lhs,sense,rhs)
            #     print()
            
            S = set()
            for j in range(len(cand)):
                if vars_x[j].X > 0.5:
                    S.add(j)

            if len(S) >= k:
                for key, value in dict_cand.items():
                    if value in S:
                        sol.append(key)
                div_sol = utils.div_subset(V, sol, dist)
                break
            else:
                div = div * (1.0 - EPS)

        except gp.GurobiError as error:
            print("Error code " + str(error))
            exit(0)

        except AttributeError:
            div = div * (1.0 - EPS)
            print(f"ILP Error. Decreasing diversity threshold to {div}")

    return graphs