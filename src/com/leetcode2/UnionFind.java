package com.leetcode2;

public class UnionFind {
	int[] ids;
	int count;
	
	public UnionFind(int count){
		this.count=count;
		ids=new int[count];
		for(int i=0;i<count;i++){
			ids[i]=i;
		}
	}
	
	public int find(int i){
		return ids[i];
	}
	
	public boolean union(int m, int n){
		int src=ids[m];
		int dst=ids[n];
		
		if(src==dst)
			return false;
		for(int i=0;i<ids.length;i++){
			if(ids[i]==src)
				ids[i]=dst;
		}
		count--;
		return true;
		
	}
	
	public boolean connected(int m, int n){
		return ids[m]==ids[n];
	}
	
	public int count(){
		return count;
	}

}
