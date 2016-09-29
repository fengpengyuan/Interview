package com.leetcode2;

public class Tuple implements Comparable<Tuple>{
	int r, c, v;
	public Tuple(int r, int c, int v){
		this.r=r;
		this.c=c;
		this.v=v;
	}
	@Override
	public int compareTo(Tuple t) {
		// TODO Auto-generated method stub
		return this.v - t.v;
	}

}
