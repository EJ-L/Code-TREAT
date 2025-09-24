def get_focal_class_other_method(focal_class_other_methods, template_category, template_id):
    rt = ''
    if len(focal_class_other_methods) != 0:
        if len(focal_class_other_methods) == 1:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += 'This is the method defined in the class: `%s`\n' % focal_class_other_methods[
                        0].strip()
                    pass
                elif template_id == 2:
                    rt += 'The class defines the following method: `%s`\n' % focal_class_other_methods[
                        0].strip()
                    pass
                elif template_id == 3:
                    rt += 'Another method implemented in the class is shown below to provide additional context:\n `%s`\n' % focal_class_other_methods[
                        0].strip()
                    pass
        else:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += "These are the methods defined in the class:\n```\n%s\n```\n" % (
                    '\n'.join(focal_class_other_methods))
                elif template_id == 2:
                    rt += "The class defines the following methods:\n```\n%s\n```\n" % (
                    '\n'.join(focal_class_other_methods))
                    pass
                elif template_id == 3:
                    rt += "Other methods implemented in the class are shown below to provide additional context:\n```\n%s\n```\n" % ('\n'.join(focal_class_other_methods))
                    pass
    return rt

def get_parameters(param_names, template_category, template_id):
    '''
    add parameter names into the prompt
    :param param_names: parameter names
    :param formatting: natural or comment
    :return: constructed prompt component
    '''

    rt = ''
    if len(param_names) == 1:
        if template_category == ['direct']:
            if template_id == 1:
                rt += 'The parameter of the method is `%s`.\n' % param_names[0]
                pass
            elif template_id == 2:
                rt += "The method takes a parameter named `%s`.\n" % param_names[0]
                pass
            elif template_id == 3:
                rt += "The method accepts a paramter: `%s`.\n" % param_names[0]
                pass
        pass
    elif len(param_names) != 0:
        if template_category == ['direct']:
            if template_id == 1:
                rt += 'There are %d parameters in the method, namely %s.\n' % (
                    len(param_names),
                    ', '.join([
                        '`' + param_str + '`' for param_str in param_names
                    ])
                )
            elif template_id == 2:
                rt += 'The method takes %d parameters, specifically %s.\n' % (
                    len(param_names),
                    ', '.join([
                        '`' + param_str + '`' for param_str in param_names
                    ])
                )
                pass
            elif template_id == 3:
                rt += 'This method accepts the following %d parameters: %s.\n' % (
                    len(param_names),
                    ', '.join([
                        '`' + param_str + '`' for param_str in param_names
                    ])
                )
                pass
        
    return rt

def get_parameter_classes(param_classes, template_category, template_id):
    rt = ''

    if len(param_classes) != 0:
        if len(param_classes) == 1:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += 'The parameters come from class `%s`.\n' % param_classes[0]
                    pass
                elif template_id == 2:
                    rt += 'The parameters originate from class `%s`.\n' % param_classes[0]
                    pass
                elif template_id == 3:
                    rt += "The parameters are defined within the class `%s`.\n" % param_classes[0]
                    pass
        else:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += 'The parameters come from classes: %s.\n' % (
                        ', '.join([
                            '`' + class_sig + '`' for class_sig in param_classes
                        ])
                    )
                elif template_id == 2:
                    rt += 'The parameters originate from classes: %s.\n' % (
                        ', '.join([
                            '`' + class_sig + '`' for class_sig in param_classes
                        ])
                    )
                    pass
                elif template_id == 3:
                    rt += 'The parameters are defined within the following classes: %s.\n' % (
                        ', '.join([
                            '`' + class_sig + '`' for class_sig in param_classes
                        ])
                    )
                    pass
    return rt

def get_parameter_class_constructors(param_class_constructors, template_category, template_id):
    rt = ''
    param_class_constructors = list(set(param_class_constructors))
    if len(param_class_constructors) == 1:
        if '|' in param_class_constructors[0]:
            param_class_constructors = param_class_constructors[0].split('|')

    if len(param_class_constructors) != 0:
        if len(param_class_constructors) == 1:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += "The constructor of the class to which the parameters of the method belong is:`%s`\n" % \
                    param_class_constructors[0]
                elif template_id == 2:
                    rt += "The class from which the parameters originate have the following constructor:`%s`\n" % \
                    param_class_constructors[0]
                    pass
                elif template_id == 3:
                    rt += "This is the constructor declared in the class associated with the method's parameters:\n"
                    pass
        else:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += ("The constructors of the class to which the parameters of the method belong are:\n"
                            "```\n"
                            "%s\n```\n ") % ('\n'.join(param_class_constructors))
                elif template_id == 2:
                    rt += ("The class from which the parameters originate have the following constructors:\n"
                            "```\n"
                            "%s\n```\n ") % ('\n'.join(param_class_constructors))
                    pass
                elif template_id == 3:
                    rt += ("These are the constructors declared in the class associated with the method's parameters:\n"
                        "```\n"
                        "%s\n```\n") % ('\n'.join(param_class_constructors))
                    pass

    return rt

def get_focal_class_constructor(focal_class_constructor, focal_class_signature, template_category, template_id):
    # Focal Class Constructor
    rt = ''
    if len(focal_class_constructor) > 1:
        if template_category == ['direct']:
            if template_id == 1:
                rt += "The focal method belongs to the class `%s`, and the constructors are:\n```\n %s \n```\n" % (
                    focal_class_signature, '\n'.join(focal_class_constructor))
            elif template_id == 2:
                rt += "The focal method is defined in the class %s, which has the following constructors:\n```\n %s \n```\n" % (focal_class_signature, '\n'.join(focal_class_constructor))
                pass
            elif template_id == 3:
                rt += "Within the class `%s`, where the focal method is implemented, the constructors are:\n```\n%s\n```\n" % (focal_class_signature, '\n'.join(focal_class_constructor))
    elif len(focal_class_constructor) == 1:
        if template_category == ['direct']:
            if template_id == 1:
                rt += "The focal method belongs to the class `%s`, and the constructor is `%s`\n" % (
                    focal_class_signature, '\n'.join(focal_class_constructor))
            elif template_id == 2:
                rt += "The focal method is defined in the class `%s`, which has the following constructor: `%s`\n" % (
                    focal_class_signature, '\n'.join(focal_class_constructor))
                pass
            elif template_id == 3:
                rt += "Within the class `%s`, where the focal method is implemented, the constructor is `%s`\n" % (
                    focal_class_signature, '\n'.join(focal_class_constructor))
    return rt

def get_focal_class_field(focal_class_fields, template_category, template_id):
    rt = ''

    if len(focal_class_fields) != 0:
        if isinstance(focal_class_fields[0], dict):
            focal_class_fields = [x['declaration_text'] for x in focal_class_fields]
        if len(focal_class_fields) == 1:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += 'This is the field defined in the class: `%s`.\n' % (
                        '\n'.join(focal_class_fields))
                    pass
                elif template_id == 2:
                    rt += 'The class defines the field: `%s`.\n' % (
                        '\n'.join(focal_class_fields))
                    pass
                elif template_id == 3:
                    rt += "This class declared the field: `%s`\n" % (
                        '\n'.join(focal_class_fields))
                    pass
        else:
            if template_category == ['direct']:
                if template_id == 1:
                    rt += "These are the fields defined in the class:\n```\n%s\n```\n" % (
                        '\n'.join(focal_class_fields))
                    pass
                elif template_id == 2:
                    rt += "The class defines the following fields:\n```\n%s\n```\n" % (
                        '\n'.join(focal_class_fields))
                    pass
                elif template_id == 3:
                    rt += "This is a list of fields declared in the class:\n```\n%s\n```\n" % (
                        '\n'.join(focal_class_fields))
                    pass

    return rt

def get_focal_class_other_method(focal_class_other_methods, template_category, template_id):
        rt = ''

        if len(focal_class_other_methods) != 0:
            if len(focal_class_other_methods) == 1:
                if template_category == ['direct']:
                    if template_id == 1:
                        rt += 'This is the method defined in the class: `%s`\n' % focal_class_other_methods[
                            0].strip()
                        pass
                    elif template_id == 2:
                        rt += 'The following method is defined in the class: `%s`\n' % focal_class_other_methods[
                            0].strip()
                        pass
                    elif template_id == 3:
                        rt += 'The class also contain the following method: `%s`\n' % focal_class_other_methods[
                            0].strip()
                        pass
            else:
                if template_category == ['direct']:
                    if template_id == 1:
                        rt += "These are the methods defined in the class:\n```\n%s\n```\n" % (
                            '\n'.join(focal_class_other_methods))
                        pass
                    elif template_id == 2:
                        rt += "The following methods are defined in the class:\n```\n%s\n```\n" % (
                            '\n'.join(focal_class_other_methods))
                        pass
                    elif template_id == 3:
                        rt += "The class also contains the following methods:\n```\n%s\n```\n" % (
                            '\n'.join(focal_class_other_methods))
                        pass
        return rt