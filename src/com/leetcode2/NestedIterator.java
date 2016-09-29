package com.leetcode2;

import java.util.List;
import java.util.Iterator;
import java.util.Stack;

public class NestedIterator implements Iterator<Integer>{
	
	Stack<NestedInteger> stk;
    public NestedIterator(List<NestedInteger> nestedList) {
        stk=new Stack<NestedInteger>();
        for(int i=nestedList.size()-1;i>=0;i--){
            stk.push(nestedList.get(i));
        }
    }

	@Override
	public boolean hasNext() {
		while(!stk.isEmpty()){
			NestedInteger cur=stk.peek();
			if(cur.isInteger())
				return true;
			// if not integer, flatten the list and push in stk again
			stk.pop();
			for(int i=cur.getList().size()-1;i>=0;i--){
				stk.push(cur.getList().get(i));
			}
		}
		return false;
	}

	@Override
	public Integer next() {
		return stk.pop().getInteger();
	}

	@Override
	public void remove() {
		// TODO Auto-generated method stub
		
	}

}
