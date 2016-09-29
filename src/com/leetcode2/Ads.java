package com.leetcode2;

public class Ads {
	int start;
	int end;
	int profit;
	
	public Ads(){
		
	}
	
	public Ads(int start, int end, int profit){
		this.start=start;
		this.end=end;
		this.profit=profit;
	}
	
	public String toString() {
		return "("+start+", "+end+")";
	}
}
