package com.leetcode2;

public class Process {
	int pid;
	int start;
	
	public Process(int pid, int start){
		this.pid=pid;
		this.start=start;
	}
	
	public String toString(){
		return pid+": "+start;
	}
}
