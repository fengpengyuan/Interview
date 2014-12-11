package com.leetcode;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;

public class Graph {
	int V;
	List<Integer>[] adj;
	
	@SuppressWarnings("unchecked")
	public Graph(int V){
		this.V=V;
		adj=(List<Integer>[]) new ArrayList<?>[V];
		for(int v=0;v<V;v++)
			adj[v]=new ArrayList<Integer>();
	}
	
	public void addEdge(int v, int w){// add w to v's list. because this is DAG, otherwise, add adj[w].add(v), bi-directional
		adj[v].add(w);
	}
	
	public void topologicalSort(){
		Stack<Integer> stk=new Stack<Integer>();
		boolean[] visited=new boolean[V];
		int[] topoNum={V};
		for(int v=0;v<V;v++){
			if(!visited[v])
				topologicalSortUtil(v, visited, stk, topoNum);
		}
		while(!stk.isEmpty()){
			System.out.print((char)('a'+stk.pop())+" ");
		}
		System.out.println();
	}
	
	public void topologicalSortUtil(int v, boolean[] visited, Stack<Integer> stk, int[] topoNum){
		visited[v]=true;
		List<Integer> adjacency=adj[v];
		for(int i=0;i<adjacency.size();i++){
			if(!visited[adjacency.get(i)])
				topologicalSortUtil(adjacency.get(i),visited, stk, topoNum);
		}
		System.out.println((char)('a'+v)+" is "+ topoNum[0]);// reverse order
		topoNum[0]--;
		stk.push(v);
	}
	
	public static void findLanguageOrder(String[] words){
		Set<Character> set=new HashSet<Character>();
		for(String word:words){
			for(int i=0;i<word.length();i++)
				set.add(word.charAt(i));
		}
		int V=set.size();
		Graph g=new Graph(V);
		
		for(int i=0;i<words.length-1;i++){
			String word1=words[i];
			String word2=words[i+1];
			for(int j=0;j<Math.min(word1.length(), word2.length());j++){
				char c1=word1.charAt(j);
				char c2=word2.charAt(j);
				if(c1!=c2){
					g.addEdge(c1-'a', c2-'a');
					break;
				}
			}
		}
		g.topologicalSort();
	}
	
	public static void main(String[] arg){
		
		String words1[] = {"caa", "aaa", "aab"};
		String words2[] = {"baa", "abcd", "abca", "cab", "cad"};
		String words3[] = {"c", "cac", "cb", "bcc", "ba"};
//		String words4[] = {"caa", "aaa", "aab"};
		findLanguageOrder(words1);
		findLanguageOrder(words2);
		findLanguageOrder(words3);
	}
}
