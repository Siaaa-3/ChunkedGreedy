#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>
#include <algorithm>
#include <boost/range/algorithm/set_algorithm.hpp>
#include <boost/range/algorithm.hpp>
#include <vector>
#include <map>
#include "edge_list.hpp"
#include <deque>
#include <random>

#include <stack>
#include <queue>
#include <climits>
#include <unordered_set>

using rabbit_order::vint;
typedef std::vector<std::vector<std::pair<vint, float>>> adjacency_list;
std::map<vint, std::vector<vint>> all_vertex_nbrs;
const int row_window = 16;

template<typename RandomAccessRange>
adjacency_list make_adj_list(const vint n, const RandomAccessRange& es) {
  using std::get;

  // Symmetrize the edge list and remove self-loops simultaneously
  // 如果存在边 (u, v, w)，则添加边 (v, u, w)。同时，移除自环（即 u == v 的边）。
  std::vector<edge_list::edge> ss(boost::size(es) * 2);
  #pragma omp parallel for
  for (size_t i = 0; i < boost::size(es); ++i) {
    auto& e = es[i];
    if (get<0>(e) != get<1>(e)) {
      ss[i * 2    ] = std::make_tuple(get<0>(e), get<1>(e), get<2>(e));
      ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), get<2>(e));
    } else {
      // Insert zero-weight edges instead of loops; they are ignored in making
      // an adjacency list
      ss[i * 2    ] = std::make_tuple(0, 0, 0.0f);
      ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
    }
  }

  // Sort the edges
  __gnu_parallel::sort(ss.begin(), ss.end());

  // Convert to an adjacency list
  adjacency_list adj(n);
  #pragma omp parallel
  {
    // Advance iterators to a boundary of a source vertex
    const auto adv = [](auto it, const auto first, const auto last) {
      while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
        ++it;
      return it;
    };

    // Compute an iterator range assigned to this thread
    const int    p      = omp_get_max_threads();
    const size_t t      = static_cast<size_t>(omp_get_thread_num());
    const size_t ifirst = ss.size() / p * (t)   + std::min(t,   ss.size() % p);
    const size_t ilast  = ss.size() / p * (t+1) + std::min(t+1, ss.size() % p);
    auto         it     = adv(ss.begin() + ifirst, ss.begin(), ss.end());
    const auto   last   = adv(ss.begin() + ilast,  ss.begin(), ss.end());

    // Reduce edges and store them in std::vector
    while (it != last) {
      const vint s = get<0>(*it);

      // Obtain an upper bound of degree and reserve memory
      const auto maxdeg = 
          std::find_if(it, last, [s](auto& x) {return get<0>(x) != s;}) - it;
      adj[s].reserve(maxdeg);

      while (it != last && get<0>(*it) == s) {
        const vint t = get<1>(*it);
        float      w = 0.0;
        while (it != last && get<0>(*it) == s && get<1>(*it) == t)
          w += get<2>(*it++);
        if (w > 0.0)
          adj[s].push_back({t, w});
      }

      // The actual degree can be smaller than the upper bound
      adj[s].shrink_to_fit();
    }
  }

  return adj;
}

vint count_unused_id(const vint n, const std::vector<edge_list::edge>& edges) {
  std::vector<char> appears(n);
  for (size_t i = 0; i < edges.size(); ++i) {
    appears[std::get<0>(edges[i])] = true;
    appears[std::get<1>(edges[i])] = true;
  }
  return static_cast<vint>(boost::count(appears, false));
}

adjacency_list read_graph(const std::string& graphpath) {
  const auto edges = edge_list::read(graphpath);  // return std::vector<edge>

  // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
  const auto n =
      boost::accumulate(edges, static_cast<vint>(0), [](vint s, auto& e) {  // edges: 要累加的范围；static_cast<vint>(0) 累加的初始值； s:当前累加值，e:范围内的元素
          return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);});  // get<0> 边的起点编号，get<1> 边的终点编号

  if (const size_t c = count_unused_id(n, edges)) {   // 统计没有使用的节点，如果节点度数为 0 则报错
    std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
              << " (zero-degree vertices or noncontiguous IDs?)\n";
  }

  return make_adj_list(n, edges);
}

std::vector<vint> vertex_nbrs(const adjacency_list& adj, vint vertex_id) { // 找到某个节点的所有邻接点
  std::vector<vint> nbrs;
  std::vector<std::pair<vint, float>> vertex_nbrs;
  vertex_nbrs = adj[vertex_id];
  for(auto v: vertex_nbrs) {
    nbrs.push_back(std::get<0>(v));
  }
  return nbrs;
}

std::map<vint, std::vector<vint>> comm_map;
std::vector<vint> v2comm;

void get_comm_map(const adjacency_list& adj) {
'''
输入:全图,adjacency_list类型(std::vector<std::vector<std::pair<vint, float>>>)
输出：填充三个全局数据结构：
● comm_map:社区ID → 节点列表的映射,map<vint, std::vector<vint>>,key是社区ID,value是社区内的节点列表
● v2comm:节点ID → 社区ID的映射,vector<vint>,index是节点ID,value是社区ID
● all_vertex_nbrs:每个节点的邻居列表,map<vint, std::vector<vint>>,key是节点ID,value是节点的邻居列表
'''  
  // 创建深拷贝
  adjacency_list _adj;
  _adj.reserve(adj.size());
  for (size_t i = 0; i < adj.size(); ++i) {
    const auto& vertex_edges = adj[i];
    
    std::vector<std::pair<vint, float>> edges;
    edges.reserve(vertex_edges.size());
    for (const auto& edge : vertex_edges) {
      edges.push_back(edge);
    }
    _adj.push_back(std::move(edges));
  }
  
  //--------------------------------------------
  auto        g = rabbit_order::aggregate(std::move(_adj));
  // const auto  p = rabbit_order::compute_perm(g);

  v2comm.assign(adj.size(), -1);

  for (vint v = 0; v < adj.size(); ++v) {
    if (v < g.n()) {
      vint comm_id = rabbit_order::trace_com(v, &g);
      comm_map[comm_id].push_back(v);
      v2comm[v] = comm_id;
    }
    all_vertex_nbrs[v] = vertex_nbrs(adj, v);
  }
}

vint pick_comm(const adjacency_list& adj) {
  vint next_comm = comm_map.begin()->first;
  vint degree = INT_MAX;
  for(auto& item: comm_map) {
    vint comm_id = item.first;
    std::vector<vint> vertexs = item.second;
    for(vint& v: vertexs) {
      vint v_degree = static_cast<vint>(boost::size(adj[v]));
      if(v_degree <= degree) {
        degree = v_degree;
        next_comm = comm_id;
      }
    }
  }
  return next_comm;
}

std::vector<vint> nbr_comm_ids(vint present_vertex_id) {
'''
输入:上一个排序完成的社区id(即上一个社区的随机一个顶点id)
输出:输入顶点的所有邻居的所属社区
'''
  std::vector<vint> nbrs_comm_ids;
  std::vector<vint> present_nbrs = all_vertex_nbrs[present_vertex_id];

  // std::cerr << "Find nbr comm ids...\n";
  //--------------------------------------------
  vint present_comm_id = v2comm[present_vertex_id];
  for(const vint& pnbr: present_nbrs) {
    vint nbr_comm_id = v2comm[pnbr];
    if(nbr_comm_id != -1 && nbr_comm_id != present_comm_id) {
      nbrs_comm_ids.push_back(nbr_comm_id);   // 放入社区的编号，之后从该社区的 map 中找到社区的所有节点，求公共邻居
    }
  }
  return nbrs_comm_ids;
}

void sort_comm(std::vector<vint>& comm_vertex, std::vector<vint>& nbrs, vint present_vid, std::deque<vint>& res, bool is_first = true) {
'''
输入:comm_vertex,社区内所有节点
输入:nbrs,当前社区根节点的邻居
'''

  vint next_vid;
  vint p_comm_id = v2comm[present_vid];
  res.push_back(present_vid);
  comm_vertex.erase(boost::remove_if(comm_vertex, [&present_vid](vint v) { return v == present_vid;}), comm_vertex.end());
  while(!comm_vertex.empty()) {
    nbrs = all_vertex_nbrs[present_vid];
    vint cnt_max_inner_comm_v = static_cast<vint>(0);
    // #pragma omp parallel for ordered
    for(const vint& v: comm_vertex) {
      std::vector<vint> two_order_nbrs = all_vertex_nbrs[v];
      std::vector<vint> intersection;
      boost::set_intersection(nbrs, two_order_nbrs, std::back_inserter(intersection));
      vint cnt_inner_comm_v = 0;
      for(vint inner_v: intersection) {
        if(v2comm[inner_v] == p_comm_id) {
          ++cnt_inner_comm_v;
        }
      }
      // #pragma omp ordered
      if(cnt_inner_comm_v >= cnt_max_inner_comm_v) {
        cnt_max_inner_comm_v = cnt_inner_comm_v;
        next_vid = v;
      }
      // std::cout << "v is : " << v << " present_vid is : " << present_vid << " cnt_inner_comm_v is : " << cnt_inner_comm_v << std::endl;
      two_order_nbrs.clear(); two_order_nbrs.shrink_to_fit();
      intersection.clear(); intersection.shrink_to_fit();
    }
    if(next_vid == present_vid) {
      std::cerr << " something goes wrong with sorting... " << std::endl;
      next_vid = comm_vertex[0];
    }
    if(is_first) res.push_front(next_vid);
    else res.push_back(next_vid);
    comm_vertex.erase(boost::remove_if(comm_vertex, [&next_vid](vint v) { return v == next_vid;}), comm_vertex.end());
    present_vid = next_vid;
  }
}


// 整个重排序的入口
// 输入：全图，adjacency_list类型（std::vector<std::vector<std::pair<vint, float>>>）
std::deque<vint> re2order_vertex(const adjacency_list& adj) {
  std::deque<vint> res; 
  std::vector<vint> nbrs;
  std::deque<vint> independent_comm_vertex;
  vint present_vid;
  get_comm_map(adj);

  // // 打印各个社区内的节点数
  // std::cout << "各个社区内的节点数：" << std::endl;
  // for (const auto& pair : comm_map) {
  //   vint comm_id = pair.first;
  //   const std::vector<vint>& vertices = pair.second;
  //   std::cout << "社区 ID: " << comm_id << ", 节点数: " << vertices.size() << std::endl;
  // }
  // return res;

  vint comm_id = comm_map.begin()->first; // 社区id（社区根节点，根节点并不代表度数最大，就是社区内随机的一个节点）
  nbrs = all_vertex_nbrs[comm_id];        // 社区根节点的邻居
  std::vector<vint> comm_vertex = comm_map[comm_id]; // 社区内的所有节点

  present_vid = comm_id;  // 随机选了一个顶点作为排序的起点

  sort_comm(comm_vertex, nbrs, present_vid, res);
  // -------------------------

  comm_map.erase(comm_id);    // 删除已经排序的社区
  std::vector<vint> vec_comms;

  // std::cout<<"next_vid :" << next_vid << std::endl;
  // 排序其他社区
  vint last_vid = comm_id;  // 上一个社区的id
  // option1: 找到下一个社区，按照度数排序，返回度数最小的社区，在这个社区中找到和 last_vid 二阶邻居节点最多的节点
  // option2: 直接查找所有剩下社区中和 last_vid 二阶邻居最多的节点 （这种方式比较好）
    // 如果剩下的社区没有还未被排序的，再找度数最小的社区从头开始
  while(!comm_map.empty()) {
    // std::cerr << "Find next comm id ..." << std::endl;
    vec_comms = nbr_comm_ids(last_vid);   // 找到当前节点相邻的社区
    if(vec_comms.empty()) { // vec_comms 为空：1. 当前节点不和其他社区相邻，是社区的内部节点，需要挑选下一个社区（按照度数）；2. 当前节点是独立节点？
      // 直接随便挑选一个社区进行排序，性能应该影响不大
      // comm_id = mp.begin()->first;  // 下一个被排序的社区
      comm_id = pick_comm(adj);
      comm_vertex = comm_map[comm_id];
      if(boost::size(comm_vertex) == 1) {   // 当前社区是独立节点
        if(boost::size(adj[comm_id]) == 1)  independent_comm_vertex.push_front(comm_id);
        else independent_comm_vertex.push_back(comm_id);
        comm_map.erase(comm_id);
        last_vid = comm_id;
        continue;
      }
      present_vid = comm_id;
      std::deque<vint> tmp_res;
      sort_comm(comm_vertex, nbrs, present_vid, tmp_res);
      for(vint v: tmp_res) {
        res.push_back(v);
      }
      comm_map.erase(comm_id);
      last_vid = comm_id;
    } else { // vec_comms 不为空，找到相邻社区中二阶邻居最多的社区进行排序
      bool diff_flag = false;
      nbrs = all_vertex_nbrs[last_vid]; // last_vid的邻居，last_vid是上一个社区的comm_id（排序时第一个加入res）
      for(const vint& c_id: vec_comms) { // 找二阶邻居最多的社区
        if(comm_map.find(c_id) == comm_map.end()) {
          continue;
        }
        diff_flag = true;
        comm_vertex = comm_map[c_id]; // 邻居社区的所有节点
        vint cnt_max_inner_comm_v = static_cast<vint>(0);
        // #pragma omp parallel for ordered
        for(const vint& v: comm_vertex) { // 获取二阶邻居：v和last_vid的交集，并且交集的顶点属于v的社区
          std::vector<vint> two_order_nbrs = all_vertex_nbrs[v];
          std::vector<vint> intersection;
          boost::set_intersection(nbrs, two_order_nbrs, std::back_inserter(intersection));
          vint cnt_inner_comm_v = 0;
          for(vint inner_v: intersection) {
            if(v2comm[inner_v] == c_id) {
              ++cnt_inner_comm_v;
            }
          }
          // #pragma omp ordered
          if(cnt_inner_comm_v >= cnt_max_inner_comm_v) {
            cnt_max_inner_comm_v = cnt_inner_comm_v;
            comm_id = c_id;  // 遍历所有邻居社区，找到最大的二阶邻居数量最多的那个顶点所在的社区id
            present_vid = v; // 遍历所有邻居社区，找到最大的二阶邻居数量最多的那个顶点id
          }
          two_order_nbrs.clear(); two_order_nbrs.shrink_to_fit();
          intersection.clear(); intersection.shrink_to_fit();
        }
      }
      if(!diff_flag) {    // 没有相邻还没被排序的社区，需要再找一个还未排序的社区开始排序
        // comm_id = mp.begin()->first;  // 下一个被排序的社区
        comm_id = pick_comm(adj);
        comm_vertex = comm_map[comm_id];
        if(boost::size(comm_vertex) == 1) {   // 当前社区是独立节点
          if(boost::size(adj[comm_id]) == 1)  independent_comm_vertex.push_front(comm_id);
          else independent_comm_vertex.push_back(comm_id);
          comm_map.erase(comm_id);
          last_vid = comm_id;
          continue;
        }
        present_vid = comm_id;
        std::deque<vint> tmp_res;
        sort_comm(comm_vertex, nbrs, present_vid, tmp_res);
        for(vint v: tmp_res) {
          res.push_back(v);
        }
        comm_map.erase(comm_id);
        last_vid = comm_id;
      } else {
        comm_vertex = comm_map[comm_id]; // 二阶邻居数量最大的社区的所有顶点
        if(boost::size(comm_vertex) == 1) {
          if(boost::size(adj[comm_id]) == 1)  independent_comm_vertex.push_front(comm_id);
          else independent_comm_vertex.push_back(comm_id);
          comm_map.erase(comm_id);
          last_vid = comm_id;
          continue;
        }
        sort_comm(comm_vertex, nbrs, present_vid, res, false);
        comm_map.erase(comm_id);
        last_vid = present_vid;
      }
    }
  }
  
  // __gnu_parallel::sort(independent_comm_vertex.begin(), independent_comm_vertex.end());
  for(const vint& v: independent_comm_vertex) {
    res.push_back(v);
  }
  
  std::vector<bool> processed(adj.size(), false);
  for(const vint& v : res) {
      if (v >= 0 && v < adj.size()) {
        processed[v] = true;
      }
  }

  for(vint v = 0; v < adj.size(); ++v) {
      if(!processed[v]) {
          res.push_back(v);
      }
  }

  // for(auto item: res) {
  //   std::cout << item << " " << std::endl;
  // }
  FILE* file = fopen("my_order.txt", "w");
  if(file) {
    for(auto item: res) {
      // std::cout << item << " " << std::endl;
      fprintf(file, "%d\n", item);
    }
  } else {
      perror("Error opening file.");
  }
  // std::cout << "res.size() is : " << boost::size(res) << std::endl;
  return res;
}

// 这里的rabbit_reorder是为了坐rabbit order的对比实验，并没有在acc中调用
// std::unique_ptr<vint[]> rabbit_reorder(adjacency_list adj) {
//   const double tstart = rabbit_order::now_sec();
//   //--------------------------------------------
//   const auto g = rabbit_order::aggregate(std::move(adj));
//   std::unique_ptr<vint[]> p = rabbit_order::compute_perm(g);
//   //--------------------------------------------
//   std::cerr << "Runtime for permutation generation [sec]: "
//             << rabbit_order::now_sec() - tstart << std::endl;

//   // Print the result
//   // std::copy(&p[0], &p[g.n()], std::ostream_iterator<vint>(std::cout, "\n"));
//   return p;
// }

std::vector<vint> handle_vertex_remap(std::deque<vint> new2ori) {
    // char line[MAX_LINE_LENGTH];
    vint num_vertex = static_cast<vint>(new2ori.size());    // origin_vertex:line, new_vertex:cnt_vertex;
    // TODO 在这里将原始矩阵 vector 和新矩阵 vector组成 pair，以便之后重排取数（空间换取时间）
    std::vector<vint> ori2new(num_vertex);
    // while(fgets(line, MAX_LINE_LENGTH, file) != NULL) {
    //     printf("Line read: %s", line);
    //     ori2new[static_cast<vint>(std::stoi(line))] = cnt_vertex;
    //     ++cnt_vertex;
    // }
    for(vint i = 0; i < num_vertex; ++i) {
        if (new2ori[i] < 0 || new2ori[i] >= num_vertex) {
            fprintf(stderr, "FATAL ERROR in handle_vertex_remap: new2ori[%d] = %d, which is out of bounds for size %d.\n", i, new2ori[i], num_vertex);
            abort();
        }
        ori2new[new2ori[i]] = i;
    }
    
    // if(cnt_vertex != num_vertex) {
    //     printf("[WARNING]: VERTEX NUM DOES NOT MATCH. PLEASE RECHECK.\n");
    // }
    return ori2new;
}
