import java.util.Stack;

public class Solution {
  //保证每次都有一个最小值进去   这样  可以与压入的数据保持同步
  private Stack<Integer> dataStack;
  private Stack<Integer> minStack;

  public Solution()
  {
    this.dataStack = new Stack<>();
    this.minStack = new Stack<>();
  }

  public void push(int node) {
    dataStack.push(node);
    if (minStack.isEmpty() || minStack.peek() != null && node < minStack.peek()) {
      minStack.push(node);
    } else {
      minStack.push(minStack.peek());
    }
  }

  public void pop() {
    int item = dataStack.pop();
    if (minStack.size() > 0) {
      minStack.pop();
    }
  }

  public int top() {
    return dataStack.peek();
  }

  public int min() {
    return minStack.peek();
  }
}