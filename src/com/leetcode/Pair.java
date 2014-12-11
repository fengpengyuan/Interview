package com.leetcode;

public class Pair {
	int first;
	int second;
	public Pair(int first, int second){
		this.first=first;
		this.second=second;
	}
	
	public String toString(){
		return first+" "+second;
	}
	
	public boolean equals(Object o){
		if(o==this)
			return true;
		if(o instanceof Pair){
			Pair p=(Pair) o;
			if(p.first==first&&p.second==second)
				return true;
			else
				return false;
		}
		return false;
	}
	
	public int hashCode(){
	    return (int) this.first*31+this.second;
	  }
}
