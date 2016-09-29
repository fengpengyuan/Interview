package com.google;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

public class Graph {
	int V = 26;
	List<GraphNode>[] adj;
	
	public Graph(){
		adj=new ArrayList[V];
		for(int i=0;i<V;i++){
			adj[i]=new ArrayList<GraphNode>();
		}
	}
	
	public void addEdge(char u, char v, double ratio){
		adj[u-'A'].add(new GraphNode(v, ratio));
	}
	
	public void construcGraph(Graph g, List<List<Object>> lists) {
		for(List l:lists){
			g.addEdge((Character)l.get(0), (Character)l.get(1), (Double)l.get(2)); 
			g.addEdge((Character)l.get(1), (Character)l.get(0), 1/(Double)l.get(2)); 
		}
	}
	
	public void getRatio(List<List<Character>> lists){
		for(List<Character> l:lists){
			char c1=l.get(0);
			char c2=l.get(1);
			Queue<Character> que=new LinkedList<Character>();
			que.add(c1);
			Map<Character, Double> map=new HashMap<Character, Double>();
			map.put(c1, 1.0);
			Set<Character> visited=new HashSet<Character>();
			while(!que.isEmpty()){
				char c=que.poll();
				if(c==c2){
					System.out.println(c1+", "+c2+", "+map.get(c2));
				}
				for(GraphNode node: adj[c-'A']){
					if(!visited.contains(node.c)){
						que.add(node.c);
						visited.add(node.c);
						map.put(node.c, node.ratio*map.get(c));
					}
				}
			}
		}
	}
	
	public static void main(String[] args){
		Graph g=new Graph();
		List<Object> edge1=new ArrayList<Object>();
		edge1.add('A');
		edge1.add('B');
		edge1.add(0.5);
		
		List<Object> edge2=new ArrayList<Object>();
		edge2.add('A');
		edge2.add('E');
		edge2.add(2.3);
		
		List<Object> edge3=new ArrayList<Object>();
		edge3.add('C');
		edge3.add('E');
		edge3.add(1.5);
		
		List<Object> edge4=new ArrayList<Object>();
		edge4.add('C');
		edge4.add('D');
		edge4.add(1.0);
		
		List<List<Object>> edges=new ArrayList<List<Object>> ();
		edges.add(edge1);
		edges.add(edge2);
		edges.add(edge3);
		edges.add(edge4);
		
		
		List<List<Character>> lists=new ArrayList<List<Character>>();
		
		List<Character> l1=new ArrayList<Character>();
		l1.add('C');
		l1.add('B');
		List<Character> l2=new ArrayList<Character>();
		l2.add('A');
		l2.add('D');
		
		List<Character> l3=new ArrayList<Character>();
		l3.add('B');
		l3.add('D');
		
		lists.add(l1);
		lists.add(l2);
		lists.add(l3);
		
		g.construcGraph(g, edges);
		g.getRatio(lists);
		
	}
	
}
