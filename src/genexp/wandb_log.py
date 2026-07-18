import wandb


class MetricSentinel:
    def __init__(self, name: str, val = None, call_fn: callable = None):
        self.name = name
        self._val = None
        self.call_fn = call_fn if call_fn is not None else lambda x: x
        
        if val is not None:
            self.val = val
    
    @property
    def val(self):
        return self._val
    
    @val.setter
    def val(self, val):
        self._val = val
        self.call_fn(val)

    def __iadd__(self, other):
        self.val = self.val + other
        return self
    
    def __str__(self):
        return str(self.val)

class NumericSentinel(MetricSentinel):
    def __init__(self, name: str, val = None, call_fn: callable = None):
        self.history = []
        def new_call_fn(val):
            self.history.append(val)
            call_fn(val)
        super().__init__(name, val, new_call_fn)
        
    def is_curr_max(self):
        return self.val == max(self.history)
    def is_curr_min(self):
        return self.val == min(self.history)
    def max_val(self):
        return max(self.history)
    def min_val(self):
        return min(self.history)
    def max_step(self):
        return self.history.index(self.max_val())
    def min_step(self):
        return self.history.index(self.min_val())
    
class WandbLogger:
    def __init__(self, use_wandb: bool = False, project_name: str = "default", run_name: str = "default", config: dict = {}):
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.config = config        
        self.run = wandb.init(project=self.project_name, name=self.run_name, config=self.config, mode="disabled" if not self.use_wandb else "online")
        
        self.step_metrics = {}
        
    
    def watch(self, name: str, step_metric: str = "global_step"):
        if step_metric not in self.step_metrics:
            print(f"Warning: step_metric '{step_metric}' not found in step_metrics, defaulting to logging without step.")
            self.run.define_metric(name)
        else:
            self.run.define_metric(name, step_metric=step_metric)
                
        def call_fn(val):
            di = {name: val}
            if step_metric in self.step_metrics:
                di.update({step_metric: self.step_metrics[step_metric]})
            wandb.log(di)
        return NumericSentinel(name, call_fn=call_fn)

        
    def set_step_metric(self, value: int, name: str):
        self.run.define_metric(name)  
                  
        def call_fn(val):
            self.step_metrics[name] = val
            wandb.log({name: val})
                  
        return MetricSentinel(name, val=value, call_fn=call_fn)
        
    def finish(self):
        self.run.finish()
            
    def set_image(self, name: str, step_metric: str):
        if step_metric not in self.step_metrics:
            print(f"Warning: step_metric '{step_metric}' not found in step_metrics, defaulting to logging without step.")
            self.run.define_metric(name)
        else:
            self.run.define_metric(name, step_metric=step_metric)
            
        def call_fn(ax):
            if step_metric in self.step_metrics:
                di = {
                    name: wandb.Image(ax, caption=f"Step {self.step_metrics[step_metric]}"),
                    step_metric: self.step_metrics[step_metric]
                }
            else:
                di = {name: wandb.Image(ax)}
            self.run.log(di)
            
        return MetricSentinel(name, call_fn=call_fn)
