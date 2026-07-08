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


class WandbLogger:
    def __init__(self, use_wandb: bool = False, project_name: str = "default", run_name: str = "default", config: dict = {}):
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.run = None
        
        if self.use_wandb:
            self.run = wandb.init(project=self.project_name, name=self.run_name, config=self.config)
        
        self.step_metrics = {}
        
    
    def watch(self, name: str, step_metric: str = "global_step"):
        if self.use_wandb:
            if step_metric not in self.step_metrics:
                print(f"Warning: step_metric '{step_metric}' not found in step_metrics, defaulting to logging without step.")
                self.run.define_metric(name)
            else:
                self.run.define_metric(name, step_metric=step_metric)
                    
            def call_fn(val):
                if self.use_wandb:
                    di = {name: val}
                    if step_metric in self.step_metrics:
                        di.update({step_metric: self.step_metrics[step_metric]})
                    wandb.log(di)
            return MetricSentinel(name, call_fn=call_fn)
        else:
            return MetricSentinel(name)
        
    def set_step_metric(self, value: int, name: str):
        if self.use_wandb:
            self.run.define_metric(name)  
                  
            def call_fn(val):
                self.step_metrics[name] = val
                if self.use_wandb:
                    wandb.log({name: val})
                
                    
            return MetricSentinel(name, val=value, call_fn=call_fn)
        else:
            return MetricSentinel(name, val=value)
        
    def finish(self):
        if self.use_wandb:
            self.run.finish()
            
    def set_image(self, name: str, step_metric: str):
        if self.use_wandb:
            if step_metric not in self.step_metrics:
                print(f"Warning: step_metric '{step_metric}' not found in step_metrics, defaulting to logging without step.")
                self.run.define_metric(name)
            else:
                self.run.define_metric(name, step_metric=step_metric)
            
            def call_fn(val):
                if self.use_wandb:
                    di = {name: wandb.Image(val)}
                    if step_metric in self.step_metrics:
                        di.update({step_metric: self.step_metrics[step_metric]})
                    wandb.log(di)
            return MetricSentinel(name, call_fn=call_fn)
        else:
            return MetricSentinel(name)

        
    
