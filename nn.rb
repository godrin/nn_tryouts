#!/usr/bin/env ruby

class InNode
  attr_accessor :val
  def initialize
    @val = 0
  end
  def inference
  end
  def reset
  end
  def add_error(e)
    puts "IN ERROR: #{e}"
  end
end

class Link
  attr_reader :weight, :inNode
  def initialize(inNode, weight=nil)
    @inNode = inNode
    @weight = weight||rand
  end
  def val
    @inNode.val * @weight
  end
  def adjust(delta)
    @weight += delta
  end
end

# https://medium.com/dataseries/basic-overview-of-convolutional-neural-network-cnn-4fcc7dbb4f17
def actFct(x)
  1/(1+Math.exp(-x))
end

class OutNode
  def initialize(inLinks)
    @inLinks = inLinks
    @b = 0
    @val = 0
  end
  def inference
    @val = 0
    @val = @inLinks.inject(0){|val,link|
      val + link.val
    } + @b
    @val = actFct(@val)
  end
  def val
    @val
  end
  def reset
    @error = @errorCount = 0
  end

  def add_error(error)
    @error += error
    @errorCount+=1
  end

  def back_propagate
    # WIE GEHT ES HIER WEITER?????
    @inLinks.each{|link|
      link.inNode.add_error(@error * link.weight)
    }
  end

  # https://de.wikipedia.org/wiki/Backpropagation - aer irgendwie falsch
  def adjust(learnRate)
    err = @error / @errorCount
    abl = @val*(1-@val)
    @inLinks.each{|l|l.adjust(abl*err*learnRate*@val)}
  end
end

class Network
  def initialize(inNodes, outNodes, all, layers)
    @inNodes = inNodes
    @outNodes = outNodes
    @all = all
    @layers = layers

    @all.each{|n|n.reset}
  end

  def inference(vals)
    @inNodes.each_with_index{|n,i|n.val=vals[i]}
    @all.each{|n|
      n.inference
    }
    @outNodes.map{|n|n.val}
  end

  def back_propagate(wantedOutput)
    @outNodes.each_with_index{|n,i| n.add_error(wantedOutput[i] - n.val)}
    @layers.reverse.each{|layer|
      layer.each{|node|node.back_propagate}
    }
  end

  def adjust(learnRate)
    @layers.reverse.each{|layer|
      layer.each{|node|node.adjust(learnRate)}
    }
  end

end

def create(width,height)
  inNodes = (1..width).map{||InNode.new}

  all = inNodes

  curNodes = inNodes
  layers = []
  height.times{
    curNodes = (1..width).map{||OutNode.new(curNodes.map{|n|Link.new(n)})}
    layers << curNodes
    all+=curNodes
  }

  Network.new(inNodes, curNodes, all, layers)
end

nn = create(1,1)
p nn
1000.times {
  p nn.inference([1])
  nn.back_propagate([1])
  nn.adjust(0.2)
}
