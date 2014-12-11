package com.leetcode;

public class Interval {
	int start;
	int end;
	Interval() { start = 0; end = 0; }
	Interval(int s, int e) { start = s; end = e; }
	
	public class IntervalNode{
		Interval interval;
		IntervalNode left;
		IntervalNode right;
		
		public IntervalNode(Interval interval){
			this.interval=interval;
		}
	}
	
	IntervalNode root;
	
	public boolean insert(Interval interval){
		if(root==null){
			root=new IntervalNode(interval);
			return true;
		}
		
		IntervalNode cur=root;
		while(cur!=null){
			if(cur.interval.start>interval.end){
				if(cur.left==null){
					cur.left=new IntervalNode(interval);
					return true;
				}
				cur=cur.left;
			}
			else if(cur.interval.end<interval.start){
				if(cur.right==null){
					cur.right=new IntervalNode(interval);
					return true;
				}
				cur=cur.right;
			}
			else
				return false;
		}
		return false;
	}
	
	public Interval find(int val){
		if(root==null)
			return null;
		IntervalNode node=root;
		while(node!=null){
			if(node.interval.start<=val&&val<=node.interval.end)
				return node.interval;
			else if(node.interval.start>val)
				node=node.left;
			else if(node.interval.end<val)
				node=node.right;
		}
		return null;
	}
	
	public IntervalNode delete(Interval interval){
		return deleteUtil(root, interval);
	}
	
	public IntervalNode deleteUtil(IntervalNode root, Interval interval){
		if(root==null)
			return root;
		if(root.interval.start>interval.end)
			root.left=deleteUtil(root.left,interval);
		else if(root.interval.end<interval.start)
			root.right=deleteUtil(root.right,interval);
		else{
			if(root.left==null){
				return root.right;
			}
			else if(root.right==null){
				return root.left;
			}
			IntervalNode node=minValueNode(root.right);
			root.interval=node.interval;
			root.right=deleteUtil(root.right,interval);
			
//			IntervalNode tmp = root;
//            root = minValueNode(root.right);
//            root.right = delete(root.right, interval);
//            root.left = tmp.left;
			
		}
		return root;
	}
	
	public IntervalNode minValueNode(IntervalNode node){
		IntervalNode cur=node;
		while(cur.left!=null){
			cur=cur.left;
		}
		return cur;
	}
}
