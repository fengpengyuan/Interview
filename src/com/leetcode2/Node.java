package com.leetcode2;

import java.util.ArrayList;
import java.util.List;

public class Node {
	List<Node> children;
	char val;
	int id;
	int weight;
	int parent;
	
	public Node(char val){
		this.val=val;
		children=new ArrayList<Node>();
	}
	
	public Node(int id, int parent, int weight){
		this.id=id;
		this.parent=parent;
		this.weight=weight;
	}
}
