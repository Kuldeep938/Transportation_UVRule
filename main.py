from flask import Flask, redirect, url_for, render_template, request, session, flash
import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='template')
app.secret_key = 'your_secret_key'  # replace 'your_secret_key' with a real secret key

@app.route('/', methods=['GET', 'POST'])
def index():
    default_suppliers = 3
    default_demands = 4
    default_matrix = [[4.0, 3.0, 2.0, 4.0], [2.0, 3.0, 2.0, 3.0], [1.0, 1.0, 2.0, 2.0]]
    default_supply = [23.0, 12.0, 9.0]  
    default_demand = [8.0, 16.0, 13.0, 7.0]     

    # Set default values
    if 'suppliers' not in session:
        session['suppliers'] = default_suppliers
    if 'demands' not in session:
        session['demands'] = default_demands
    if 'matrix' not in session:
        session['matrix'] = default_matrix
    if 'supply' not in session:
        session['supply'] = default_supply
    if 'demand' not in session:
        session['demand'] = default_demand
        
    if request.method == 'POST':
        suppliers = request.form.get('suppliers')
        demands = request.form.get('demands')
        matrix = request.form.get('matrix')
        supply = request.form.get('supply')
        demand = request.form.get('demand')

        if suppliers and demands and matrix and supply and demand:
            try:
                session['suppliers'] = int(suppliers)
                session['demands'] = int(demands)
                session['matrix'] = [list(map(float, row.split())) for row in matrix.split("\n")]
                session['supply'] = list(map(float, supply.split()))
                session['demand'] = list(map(float, demand.split()))
                session['idx'] = 0
            except ValueError:
                flash('Invalid input. Please check all fields.')
                return redirect(url_for('index'))

            return redirect(url_for('costmatrix'))

    return render_template('index.html', 
                           suppliers=session.get('suppliers'),
                           demands=session.get('demands'),
                           matrix="\n".join([" ".join(map(str, row)) for row in session.get('matrix', [])]),
                           supply=" ".join(map(str, session.get('supply', []))),
                           demand=" ".join(map(str, session.get('demand', []))))

@app.route('/costmatrix', methods=['GET', 'POST'])
def costmatrix():
    if request.method == 'POST':
        session['method'] = request.form.get('method')
        return redirect(url_for('resultbfs'))

    return render_template('costmatrix.html', suppliers=session.get('suppliers'), demands=session.get('demands'), matrix=session.get('matrix'), supply=session.get('supply'), demand=session.get('demand'))

def northWest(supply,demand):
    solution = [[0 for _ in range(len(demand))] for _ in range(len(supply))]
    s_idx = 0
    d_idx = 0
    supply_cpy = copy.deepcopy(supply)
    demand_cpy = copy.deepcopy(demand)
    while (s_idx < len(supply) and d_idx<len(demand)):
        if(supply_cpy[s_idx] > demand_cpy[d_idx]):
            supply_cpy[s_idx] = supply_cpy[s_idx] - demand_cpy[d_idx]
            solution[s_idx][d_idx] = demand_cpy[d_idx]
            demand_cpy[d_idx] = 0
            d_idx = d_idx + 1
        else:
            demand_cpy[d_idx] = demand_cpy[d_idx] - supply_cpy[s_idx]
            solution[s_idx][d_idx] = supply_cpy[s_idx]
            supply_cpy[s_idx] = 0
            s_idx = s_idx +1
    return solution

def minCost(supply,demand,matrix):
    Cost = copy.deepcopy(matrix)
    sources_cpy = copy.deepcopy(supply)
    destinations_cpy = copy.deepcopy(demand)
    Allocation = [[0 for _ in range(len(destinations_cpy))] for _ in range(len(sources_cpy))]
    def Travel(S, D, Max):
        if sources_cpy[S] != 0 and destinations_cpy[D] != 0:
            if sources_cpy[S] > destinations_cpy[D]:
                sources_cpy[S] -= destinations_cpy[D]
                Allocation[S][D] = destinations_cpy[D]
                destinations_cpy[D] = 0

                for i in range(len(sources_cpy)):
                    Cost[i][D] = Max + 1

            elif sources_cpy[S] < destinations_cpy[D]:
                destinations_cpy[D] -= sources_cpy[S]
                Allocation[S][D] = sources_cpy[S]
                sources_cpy[S] = 0
                for i in range(len(destinations_cpy)):
                    Cost[S][i] = Max + 1

            else:
                Allocation[S][D] = destinations_cpy[D]
                sources_cpy[S] = 0
                destinations_cpy[D] = 0

                for i in range(len(destinations_cpy)):
                    Cost[S][i] = Max + 1
                for i in range(len(sources_cpy)):
                    Cost[i][D] = Max + 1
    def Min():
        temp = Cost[0][0]
        for i in range(len(sources_cpy)):
            for j in range(len(destinations_cpy)):
                if Cost[i][j] < temp:
                    temp = Cost[i][j]
        return temp
    Max = max([max(row) for row in Cost])
    while Min() != Max + 1:
        temp = Cost[0][0]
        for i in range(len(sources_cpy)):
            for j in range(len(destinations_cpy)):
                if Cost[i][j] < temp:
                    temp = Cost[i][j]
                    S = i
                    D = j
        Travel(S, D, Max)
    return Allocation

def calculate_uv(cost_matrix, bfs, u, v): 
    while None in u or None in v:
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if bfs[i][j] != 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost_matrix[i][j] - u[i]
                    elif u[i] is None and v[j] is not None:
                        u[i] = cost_matrix[i][j] - v[j]
    return


def calculate_opportunity_costs(cost_matrix, bfs, u, v):
    opportunity_costs = []
    for i in range(len(cost_matrix)):
        opportunity_costs.append([])
        for j in range(len(cost_matrix[i])):
            if bfs[i][j] == 0:
                opportunity_costs[i].append(cost_matrix[i][j] - u[i] - v[j])
            else:
                opportunity_costs[i].append(None)
    return opportunity_costs


def find_most_negative_opportunity_cost(opportunity_costs):
    min_val = 0
    min_i = -1
    min_j = -1
    for i in range(len(opportunity_costs)):
        for j in range(len(opportunity_costs[i])):
            if opportunity_costs[i][j] is not None and opportunity_costs[i][j] < min_val:
                min_val = opportunity_costs[i][j]
                min_i = i
                min_j = j
    return min_i, min_j, min_val

def find_loop(bfs, stop, start):
    graph = {}
    sources = len(bfs)
    destinations = len(bfs[0])
    source_nodes = []
    destination_nodes = []
    for i in range(sources):
        source_nodes.append((i,0))
    for i in range(destinations):
        destination_nodes.append((i,1))
    for i in source_nodes:
        for j in destination_nodes:
            if bfs[i[0]][j[0]] != 0:
                if i in graph:
                    graph[i].append(j)
                else:
                    graph[i] = [j]
    for i in destination_nodes:
        for j in source_nodes:
            if bfs[j[0]][i[0]] != 0:
                if i in graph:
                    graph[i].append(j)
                else:
                    graph[i] = [j] 
                           
    def dfs_cycle(graph, stop, start):
        visited = set()
        stack = [((start,1), [])]

        while stack:
            (node, path) = stack.pop()
            if node not in visited:
                if node == (stop,0):
                    return path + [node]
                visited.add(node)
                for neighbor in graph[node]:
                    stack.append((neighbor, path + [node]))
        return None
    
    
    path = dfs_cycle(graph, stop, start)
    loop = [(stop,start,"+")]
    switch = 0
    
    for i in range(len(path)-1):
        if switch == 1:
            loop.append((path[i][0],path[i+1][0],"+"))
        elif switch == 0:
            loop.append((path[i+1][0],path[i][0],"-"))
        switch = 1-switch
    return loop


def update_bfs(bfs, loop):
    if  len(loop) == 0:
        return

    min_values = [bfs[i][j] for (i, j, sign) in loop if sign == "-"]

    if not min_values:  
        return

    min_val = min(min_values)
    for i, j, sign in loop:
        if sign == "+":
            bfs[i][j] += min_val
        else:
            bfs[i][j] -= min_val
        if bfs[i][j] == 0:
            bfs[i][j] = 0

def uv_method(cost_matrix, bfs):
    iteration = 0
    bfss = []
    while True:
      #  print (bfs)
        temp = copy.deepcopy(bfs)
        bfss.append(temp)
        u = [None] * len(cost_matrix)
        v = [None] * len(cost_matrix[0])
        u[0] = 0
        
        calculate_uv(cost_matrix, bfs, u, v)
      #  print(u)
      #  print(v)
        opportunity_costs = calculate_opportunity_costs(cost_matrix, bfs, u, v)
        entering_i, entering_j, min_opportunity_cost = find_most_negative_opportunity_cost(opportunity_costs)
       # print(find_most_negative_opportunity_cost(opportunity_costs))
        if min_opportunity_cost >= 0:
            break
        loop = find_loop(bfs, entering_i, entering_j)
        update_bfs(bfs, loop)
        iteration = iteration+1
    return bfss

def calCost(bfs,Cost):
    sum = 0
    for i in range(len(bfs)):
        for j in range(len(bfs[0])):
            sum = sum + bfs[i][j]*Cost[i][j]
    return sum


def network(bfs):
    supp = []
    dem = []
    allocation = []
    no_of_supp = len(bfs)
    no_of_dem = len(bfs[0])
    
    for i in range(no_of_supp):
        supp.append(('S'+str(i+1)))
    for i in range(no_of_dem):
        dem.append(('D'+str(i+1)))
    G = nx.DiGraph()
    G.add_nodes_from(supp)
    G.add_nodes_from(dem)
    for i, supplier in enumerate(supp):
        for j, destination in enumerate(dem):
            if bfs[i][j] != 0:
                 G.add_edge(supplier, destination,weight = bfs[i][j])
    pos = {}

    for i, supplier in enumerate(supp):
        pos[supplier] = (1, i+1)

    for i, destination in enumerate(dem):
        pos[destination] = (2, i+1)

    nx.draw_networkx_nodes(G, pos, node_color='skyblue')
    nx.draw_networkx_labels(G, pos)

    for edge in G.edges:
        start, end = pos[edge[0]], pos[edge[1]]
        plt.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='black', arrowstyle='-'))
    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    for (start_node, end_node), weight in edge_labels.items():
        x = (pos[start_node][0] + pos[end_node][0]) / 2
        y = (pos[start_node][1] + pos[end_node][1]) / 2
        plt.text(x - 0.1, y, weight, horizontalalignment='right', verticalalignment='center')
    x_sup = 1
    x_dem = 2
    y_sup = 1
    y_dem = 1
    for i in range(no_of_supp):
        sum = 0
        for j in range(no_of_dem):
            sum += bfs[i][j]
        plt.text(x_sup+0.04, y_sup+0.15,sum , horizontalalignment='right', verticalalignment='center')
        y_sup += 1
    
    for i in range(no_of_dem):
        sum = 0
        for j in range(no_of_supp):
            sum = sum + bfs[j][i]
        plt.text(x_dem, y_dem+0.15,sum , horizontalalignment='right', verticalalignment='center')
        y_dem += 1
   # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig('static/download.png')
    plt.close()
    #return
    #plt.axis('off')
   # plt.show()


@app.route('/resultbfs', methods=['GET', 'POST'])
def resultbfs():
    method = session.get('method')
    suppliers = session.get('suppliers')
    demands = session.get('demands')
    matrix = session.get('matrix')
    supply = session.get('supply')
    demand = session.get('demand')

    if suppliers and demands and matrix and supply and demand and method:
        if method == "North-West":
            solution = northWest(supply, demand)
            bfss = uv_method(matrix, solution)
            session['bfss'] = bfss
        elif method == "Minimum-Cost":
            solution = minCost(supply, demand,matrix)
            bfss = uv_method(matrix, solution)
            session['bfss'] = bfss
        else:
            solution = None
        message = ""
        
        idx = 0
        cost_bfs = calCost(bfss[idx],matrix)
        if(len(bfss)==1):
            message = "Initial Basic Solution is the Optimal Solution"
        else:
            message = "Initial Basic Solution"
        network(bfss[idx])
        return render_template('resultbfs.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,idx = idx , message=message, cost_bfs = cost_bfs)

    flash('Some information is missing, please fill in all the fields.')
    return redirect(url_for('index'))
@app.route('/next', methods=['GET', 'POST'])
def next():
    method = session.get('method')
    suppliers = session.get('suppliers')
    demands = session.get('demands')
    matrix = session.get('matrix')
    supply = session.get('supply')
    demand = session.get('demand')
    idx = session.get('idx')
    idx = idx + 1
    bfss = session.get('bfss')
    if(idx < 0):
            idx = 0
    if(idx>=len(bfss)):
        idx = len(bfss)-1
    session['idx'] = idx
    

    if suppliers and demands and matrix and supply and demand and method:
        message = ""
        if(idx ==0):
            message = "Inital Basic Feasible Solution"
            cost_bfs = calCost(bfss[idx],matrix)
            network(bfss[idx])
            return render_template('next.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,idx = idx , message = message, cost_bfs = cost_bfs)
        if(idx ==len(bfss)-1):
            message = "Optimal Solution Reached"
            cost_bfs = calCost(bfss[idx],matrix)
            network(bfss[idx])
            return render_template('next.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,idx = idx, message = message, cost_bfs = cost_bfs)
        message = "Basic Feasible Solution"
        cost_bfs = calCost(bfss[idx],matrix)
        network(bfss[idx])
        return render_template('next.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,idx = idx ,message = message, cost_bfs = cost_bfs)

@app.route('/prev', methods=['GET', 'POST'])
def prev():
    method = session.get('method')
    suppliers = session.get('suppliers')
    demands = session.get('demands')
    matrix = session.get('matrix')
    supply = session.get('supply')
    demand = session.get('demand')
    idx = session.get('idx')
    idx = idx -1
    bfss = session.get('bfss')
    if(idx < 0):
            idx = 0
    if(idx>=len(bfss)):
        idx = len(bfss)-1
    session['idx'] = idx
    

    if suppliers and demands and matrix and supply and demand and method:
        message = ""
        if(idx ==0):
            message = "Inital Basic Feasible Solution"
            cost_bfs = calCost(bfss[idx],matrix)
            network(bfss[idx])
            return render_template('next.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,message = message ,idx = idx, cost_bfs = cost_bfs)
        if(idx ==len(bfss)-1):
            message = "Optimal Solution Reached"
            cost_bfs = calCost(bfss[idx],matrix)
            network(bfss[idx])
            return render_template('next.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,idx = idx,message = message, cost_bfs = cost_bfs)
        message = "Basic Feasible Solution"
        cost_bfs = calCost(bfss[idx],matrix)
        network(bfss[idx])
        return render_template('prev.html', solution=bfss[idx], suppliers=suppliers, demands=demands, matrix=matrix, supply=supply, demand=demand,method = method,idx = idx,message = message, cost_bfs = cost_bfs)
    flash('Some information is missing, please fill in all the fields.')
    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True , threaded=False)
