package com.leetcode2;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Graph {
	int V;
	List<Integer>[] adj;
	
	public Graph(int v){
		this.V=v;
		adj=new ArrayList[v];
		for(int i=0;i<v;i++){
			adj[i]=new LinkedList<Integer>();
		}
	}
	
	public void addEdge(int v, int w){
		adj[v].add(w);
	}
}
