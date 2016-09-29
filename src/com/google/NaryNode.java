package com.google;

import java.util.ArrayList;
import java.util.List;

public class NaryNode {
	int val;
	List<NaryNode> children;
	
	public NaryNode(int val){
		this.val=val;
		children=new ArrayList<NaryNode>();
	}
}
