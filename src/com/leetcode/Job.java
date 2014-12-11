package com.leetcode;

public class Job {

	char id; // Job Id
	int dead; // Deadline of job
	int profit;
	
	public Job(char id, int deadline, int profit){
		this.id=id;
		this.dead=deadline;
		this.profit=profit;
	}
}
