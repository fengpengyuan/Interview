package others;

public class TrieNode {
	char c;
	int count;
	TrieNode[] children;
	boolean isComplete;
	
	public TrieNode(char c){
		this.c=c;
		this.count=1;
		this.children=new TrieNode[26];
		this.isComplete=false;
	}
}
