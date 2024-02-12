import numpy as np
from fmmd.utils_c import update_dists
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def gonzales_algorithm_c(
        set solution_idxs,
        double[:,::1] features,
        int k,
        double diversity = -np.inf,
        int lower_constraint = 1,
):
    cdef int sol_len = 0
    # if initial solution is empty add first element
    if len(solution_idxs) == 0:
        solution_idxs = set([0])
        sol_len = 1
    else:
        sol_len = len(solution_idxs)
    # node distances to candidates
    _element_distances = np.repeat(np.inf, features.shape[0])
    cdef double[::1] element_distances = _element_distances
    # update for each candidate
    cdef double sol_div = np.inf
    cdef Py_ssize_t solution_idx

    for solution_idx in solution_idxs:
        sol_div = min(sol_div, element_distances[solution_idx])
        update_dists(element_distances, features, features[solution_idx])
    
    cdef Py_ssize_t max_idx
    cdef double max_dist
    cdef double[::1] max_item
    while sol_len < k:
        max_idx = np.argmax(element_distances)
        max_dist = element_distances[max_idx]
        max_item = features[max_idx]
        sol_div = min(sol_div, max_dist)
        # print(f"\t\t{max_dist=}")
        # print(f"\t\t{sol_div=}")
        if sol_div < diversity:
            break
        # print(f"\t\tAdded item {ids[max_idx]}")
        solution_idxs.add(max_idx)
        sol_len +=1
        # update the closest distance to current candidate set
        update_dists(element_distances, features, max_item)
    # if performing for a group then check the constraint is met
    if sol_len < lower_constraint and diversity != -np.inf:
        return None, None
    else:
        return solution_idxs, sol_div


def fmmd_exact(features: np.ndarray,
    ids: np.ndarray,
    groups: np.ndarray,
    k: int,
    constraints: Dict[int,Tuple[int, int]],
):
    dists = pdist(features)
    print("Finished computing pdist")
    dists_sorted_args = np.argsort(dists)
    N = features.shape[0]
    M = len(dists)
    low = 0
    high = M
    curr = (low+high)//2

    sol = list()
    while low< high-1:
        print(low,curr,high)
        try:
            model = gp.Model("mis_" + str(curr))
            model.setParam(GRB.Param.TimeLimit, 300)

            size = [1] * N
            vars_x = model.addVars(N, vtype=GRB.BINARY, obj=size, name="x")

            model.modelSense = GRB.MAXIMIZE

            eid = 0
            for k in range(M-1,curr,-1):
                idx = dists_sorted_args[k]
                i = N - 2 - int(math.sqrt(-8*idx + 4*N*(N-1)-7)/2.0 - 0.5)
                j = idx + i + 1 - M + (N-i)*((N-i)-1)/2
                model.addConstr(vars_x[i] + vars_x[j] <= 1, "edge_" + str(eid))
                eid += 1

            expr = gp.LinExpr()
            for j in range(N):
                expr.addTerms(1, vars_x[j])
            model.addConstr(expr <= k, "size")

            unique_groups = np.unique(groups)
            grp_expr = {grp: gp.LinExpr() for grp in unique_groups}
            grp_lb = dict()
            grp_up = dict()
            for node in range(N):
                node_group = groups[node]
                grp_expr[node_group].addTerms(1, vars_x[node])
                grp_lb[node_group] = constraints[node_group][0]
                grp_up[node_group] = constraints[node_group][1]
            for grp, _expr in grp_expr.items():
                model.addConstr(_expr >= grp_lb[grp], "lb_color_" + str(grp))
                model.addConstr(_expr <= grp_up[grp], "ub_color_" + str(grp))

            model.optimize()
            S = set()
            for x, var in vars_x.items():
                if var.X > 0.5:
                    S.add(ids[x])
            if len(S) >= k:
                if len(sol) == 0:
                    sol.extend(S)
                    div_sol = compute_diversity(sol,features,ids)
                else:
                    div_S = compute_diversity(S,features,ids)
                    if div_S > div_sol:
                        sol.clear()
                        sol.extend(S)
                        div_sol = div_S
                high = curr
                curr = (low+high)//2
            else:
                low = curr
                curr = (low+high)//2
        except gp.GurobiError as error:
            print("Error code " + str(error))
            exit(0)

        except AttributeError:
            low = curr
            curr = (low + high) // 2
        
    return sol,div_sol