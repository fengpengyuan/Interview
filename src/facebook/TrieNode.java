package facebook;

import java.util.HashMap;

public class TrieNode {
	private char c;
	private HashMap<Character, TrieNode> children;
	private boolean isEnd;
	
	public TrieNode(char c){
		this.c=c;
		this.children=new HashMap<Character, TrieNode>();
		this.isEnd=false;
	}
	
	public HashMap<Character,TrieNode> getChildren() { 
		return children;
	}    
    public char getValue() { 
    	return c;
    }    
    public void setIsEnd(boolean val) {
    	isEnd = val; 
    }    
    public boolean isEnd(){     
    	return isEnd;
    }
}
