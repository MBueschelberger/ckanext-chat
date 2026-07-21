import ckan.plugins as p


def chat_submit(context, data_dict, privilege="package_update"):
    user = context.get("user")
    if user and context.get("auth_user_obj"):
        return {"success": True}
    else:
        return {"success": False}


def get_auth_functions():
    return {
        "chat_submit": chat_submit,
    }
